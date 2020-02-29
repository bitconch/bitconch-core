#include <chainbase/pinnable_mapped_file.hpp>
#include <chainbase/environment.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/anonymous_shared_memory.hpp>
#include <boost/asio/signal_set.hpp>
#include <iostream>

#ifdef __linux__
#include <sys/vfs.h>
#include <linux/magic.h>
#endif

namespace chainbase {

pinnable_mapped_file::pinnable_mapped_file(const bfs::path& dir, bool writable, uint64_t shared_file_size, bool allow_dirty,
                                          map_mode mode, std::vector<std::string> hugepage_paths) :
   _data_file_path(bfs::absolute(dir/"shared_memory.bin")),
   _database_name(dir.filename().string()),
   _writable(writable)
{
   if(shared_file_size % _db_size_multiple_requirement)
      BOOST_THROW_EXCEPTION(std::runtime_error("Database must be mulitple of " + std::to_string(_db_size_multiple_requirement) + " bytes"));
#ifndef __linux__
   if(hugepage_paths.size())
      BOOST_THROW_EXCEPTION(std::runtime_error("Hugepage support is a linux only feature"));
#endif
   if(hugepage_paths.size() && mode != locked)
      BOOST_THROW_EXCEPTION(std::runtime_error("Locked mode is required for hugepage usage"));
#ifdef _WIN32
   if(mode == locked)
      BOOST_THROW_EXCEPTION(std::runtime_error("Locked mode not supported on win32"));
#endif

   if(!_writable && !bfs::exists(_data_file_path))
      BOOST_THROW_EXCEPTION(std::runtime_error("database file not found at " + _data_file_path.string()));
   bfs::create_directories(dir);

   if(bfs::exists(_data_file_path)) {
      char header[header_size];
      std::ifstream hs(_data_file_path.generic_string(), std::ifstream::binary);
      hs.read(header, header_size);
      if(hs.fail())
         BOOST_THROW_EXCEPTION(std::runtime_error("Failed to read DB header."));

      db_header* dbheader = reinterpret_cast<db_header*>(header);
      if(dbheader->id != header_id)
         BOOST_THROW_EXCEPTION(std::runtime_error("\"" + _database_name + "\" database format not compatible with this version of chainbase."));
      if(!allow_dirty && dbheader->dirty)
         throw std::runtime_error("\"" + _database_name + "\" database dirty flag set");
      if(dbheader->environ != environment()) {
         std::cerr << "CHAINBASE: \"" << _database_name << "\" database was created with a chainbase from a different environment" << std::endl;
         std::cerr << "Current compiler environment:" << std::endl;
         std::cerr << environment();
         std::cerr << "DB created with compiler environment:" << std::endl;
         std::cerr << dbheader->environ;
         BOOST_THROW_EXCEPTION(std::runtime_error("All environment parameters must match"));
      }
   }

   segment_manager* file_mapped_segment_manager = nullptr;
   if(!bfs::exists(_data_file_path)) {
      std::ofstream ofs(_data_file_path.generic_string(), std::ofstream::trunc);
      bfs::resize_file(_data_file_path, shared_file_size);
      _file_mapped_region = bip::mapped_region(bip::file_mapping(_data_file_path.generic_string().c_str(), bip::read_write), bip::read_write);
      file_mapped_segment_manager = new ((char*)_file_mapped_region.get_address()+header_size) segment_manager(shared_file_size-header_size);
      new (_file_mapped_region.get_address()) db_header;
   }
   else if(_writable) {
         auto existing_file_size = bfs::file_size(_data_file_path);
         size_t grow = 0;
         if(shared_file_size > existing_file_size) {
            grow = shared_file_size - existing_file_size;
            bfs::resize_file(_data_file_path, shared_file_size);
         }
         _file_mapped_region = bip::mapped_region(bip::file_mapping(_data_file_path.generic_string().c_str(), bip::read_write), bip::read_write);
         file_mapped_segment_manager = reinterpret_cast<segment_manager*>((char*)_file_mapped_region.get_address()+header_size);
         if(grow)
            file_mapped_segment_manager->grow(grow);
   }
   else {
         _file_mapped_region = bip::mapped_region(bip::file_mapping(_data_file_path.generic_string().c_str(), bip::read_only), bip::read_only);
         file_mapped_segment_manager = reinterpret_cast<segment_manager*>((char*)_file_mapped_region.get_address()+header_size);
   }

   if(_writable) {
      //remove meta file created in earlier versions
      boost::system::error_code ec;
      bfs::remove(bfs::absolute(dir/"shared_memory.meta"), ec);

      _mapped_file_lock = bip::file_lock(_data_file_path.generic_string().c_str());
      if(!_mapped_file_lock.try_lock())
         BOOST_THROW_EXCEPTION(std::runtime_error("could not gain write access to the shared memory file"));

      set_mapped_file_db_dirty(true);
   }

   if(mode == mapped) {
      _segment_manager = file_mapped_segment_manager;
   }
   else {
      boost::asio::io_service sig_ios;
      boost::asio::signal_set sig_set(sig_ios, SIGINT, SIGTERM);
#ifdef SIGPIPE
      sig_set.add(SIGPIPE);
#endif
      sig_set.async_wait([](const boost::system::error_code&, int) {
         BOOST_THROW_EXCEPTION(std::runtime_error("Database load aborted"));
      });

      try {
         if(mode == heap)
            _mapped_region = bip::mapped_region(bip::anonymous_shared_memory(shared_file_size));
         else
            _mapped_region = get_huge_region(hugepage_paths);

         load_database_file(sig_ios);

         if(mode == locked) {
#ifndef _WIN32
            if(mlock(_mapped_region.get_address(), _mapped_region.get_size()))
               BOOST_THROW_EXCEPTION(std::runtime_error("Failed to mlock database \"" + _database_name + "\""));
            std::cerr << "CHAINBASE: Database \"" << _database_name << "\" has been successfully locked in memory" << std::endl;
#endif
         }
      }
      catch(...) {
         if(_writable)
            set_mapped_file_db_dirty(false);
         throw;
      }

      _segment_manager = reinterpret_cast<segment_manager*>((char*)_mapped_region.get_address()+header_size);
   }
}

bip::mapped_region pinnable_mapped_file::get_huge_region(const std::vector<std::string>& huge_paths) {
   std::map<unsigned, std::string> page_size_to_paths;
   const auto mapped_file_size = _file_mapped_region.get_size();

#ifdef __linux__
   for(const std::string& p : huge_paths) {
      struct statfs fs;
      if(statfs(p.c_str(), &fs))
         BOOST_THROW_EXCEPTION(std::runtime_error(std::string("Could not statfs() path ") + p));
      if(fs.f_type != HUGETLBFS_MAGIC)
         BOOST_THROW_EXCEPTION(std::runtime_error(p + std::string(" does not look like a hugepagefs mount")));
      page_size_to_paths[fs.f_bsize] = p;
   }
   for(auto it = page_size_to_paths.rbegin(); it != page_size_to_paths.rend(); ++it) {
      if(mapped_file_size % it->first == 0) {
         bfs::path hugepath = bfs::unique_path(bfs::path(it->second + "/%%%%%%%%%%%%%%%%%%%%%%%%%%"));
         int fd = creat(hugepath.string().c_str(), _db_permissions.get_permissions());
         if(fd < 0)
            BOOST_THROW_EXCEPTION(std::runtime_error(std::string("Could not open hugepage file in ") + it->second + ": " + std::string(strerror(errno))));
         if(ftruncate(fd, mapped_file_size))
            BOOST_THROW_EXCEPTION(std::runtime_error(std::string("Failed to grow hugepage file to specified size")));
         close(fd);
         bip::file_mapping filemap(hugepath.generic_string().c_str(), _writable ? bip::read_write : bip::read_only);
         bfs::remove(hugepath);
         std::cerr << "CHAINBASE: Database \"" << _database_name << "\" using " << it->first << " byte pages" << std::endl;
         return bip::mapped_region(filemap, _writable ? bip::read_write : bip::read_only);
      }
   }
#endif

   std::cerr << "CHAINBASE: Database \"" << _database_name << "\" not using huge pages" << std::endl;
   return bip::mapped_region(bip::anonymous_shared_memory(mapped_file_size));
}

void pinnable_mapped_file::load_database_file(boost::asio::io_service& sig_ios) {
   std::cerr << "CHAINBASE: Preloading \"" << _database_name << "\" database file, this could take a moment..." << std::endl;
   char* const src = (char*)_file_mapped_region.get_address();
   char* const dst = (char*)_mapped_region.get_address();
   size_t offset = 0;
   time_t t = time(nullptr);
   while(offset != _file_mapped_region.get_size()) {
      memcpy(dst+offset, src+offset, _db_size_multiple_requirement);
      offset += _db_size_multiple_requirement;

      if(time(nullptr) != t) {
         t = time(nullptr);
         std::cerr << "              " << offset/(_mapped_region.get_size()/100) << "% complete..." << std::endl;
      }
      sig_ios.poll();
   }
   std::cerr << "           Complete" << std::endl;
}

bool pinnable_mapped_file::all_zeros(char* data, size_t sz) {
   uint64_t* p = (uint64_t*)data;
   uint64_t* end = p+sz/sizeof(uint64_t);
   while(p != end) {
      if(*p++ != 0)
         return false;
   }
   return true;
}

void pinnable_mapped_file::save_database_file() {
   std::cerr << "CHAINBASE: Writing \"" << _database_name << "\" database file, this could take a moment..." << std::endl;
   char* src = (char*)_mapped_region.get_address();
   char* dst = (char*)_file_mapped_region.get_address();
   size_t offset = 0;
   time_t t = time(nullptr);
   while(offset != _file_mapped_region.get_size()) {
      if(!all_zeros(src+offset, _db_size_multiple_requirement))
         memcpy(dst+offset, src+offset, _db_size_multiple_requirement);
      offset += _db_size_multiple_requirement;

      if(time(nullptr) != t) {
         t = time(nullptr);
         std::cerr << "              " << offset/(_file_mapped_region.get_size()/100) << "% complete..." << std::endl;
      }
   }
   std::cerr << "           Syncing buffers..." << std::endl;
   if(_file_mapped_region.flush(0, 0, false) == false)
      std::cerr << "CHAINBASE: ERROR: syncing buffers failed" << std::endl;
   std::cerr << "           Complete" << std::endl;
}

pinnable_mapped_file::pinnable_mapped_file(pinnable_mapped_file&& o) :
   _mapped_file_lock(std::move(o._mapped_file_lock)),
   _data_file_path(std::move(o._data_file_path)),
   _database_name(std::move(o._database_name)),
   _file_mapped_region(std::move(o._file_mapped_region)),
   _mapped_region(std::move(o._mapped_region))
{
   _segment_manager = o._segment_manager;
   _writable = o._writable;
   o._writable = false; //prevent dtor from doing anything interesting
}

pinnable_mapped_file& pinnable_mapped_file::operator=(pinnable_mapped_file&& o) {
   _mapped_file_lock = std::move(o._mapped_file_lock);
   _data_file_path = std::move(o._data_file_path);
   _database_name = std::move(o._database_name);
   _file_mapped_region = std::move(o._file_mapped_region);
   _mapped_region = std::move(o._mapped_region);
   _segment_manager = o._segment_manager;
   _writable = o._writable;
   o._writable = false; //prevent dtor from doing anything interesting
   return *this;
}

pinnable_mapped_file::~pinnable_mapped_file() {
   if(_writable) {
      if(_mapped_region.get_address()) //in heap or locked mode
         save_database_file();
      else
         if(_file_mapped_region.flush(0, 0, false) == false)
            std::cerr << "CHAINBASE: ERROR: syncing buffers failed" << std::endl;
      set_mapped_file_db_dirty(false);
   }
}

void pinnable_mapped_file::set_mapped_file_db_dirty(bool dirty) {
   *((char*)_file_mapped_region.get_address()+header_dirty_bit_offset) = dirty;
   if(_file_mapped_region.flush(0, 0, false) == false)
      std::cerr << "CHAINBASE: ERROR: syncing buffers failed" << std::endl;
}

std::istream& operator>>(std::istream& in, pinnable_mapped_file::map_mode& runtime) {
   std::string s;
   in >> s;
   if (s == "mapped")
      runtime = pinnable_mapped_file::map_mode::mapped;
   else if (s == "heap")
      runtime = pinnable_mapped_file::map_mode::heap;
   else if (s == "locked")
      runtime = pinnable_mapped_file::map_mode::locked;
   else
      in.setstate(std::ios_base::failbit);
   return in;
}

std::ostream& operator<<(std::ostream& osm, pinnable_mapped_file::map_mode m) {
   if(m == pinnable_mapped_file::map_mode::mapped)
      osm << "mapped";
   else if (m == pinnable_mapped_file::map_mode::heap)
      osm << "heap";
   else if (m == pinnable_mapped_file::map_mode::locked)
      osm << "locked";

   return osm;
}

static std::string print_os(environment::os_t os) {
   switch(os) {
      case environment::OS_LINUX: return "Linux";
      case environment::OS_MACOS: return "macOS";
      case environment::OS_WINDOWS: return "Windows";
      case environment::OS_OTHER: return "Unknown";
   }
   return "error";
}
static std::string print_arch(environment::arch_t arch) {
   switch(arch) {
      case environment::ARCH_X86_64: return "x86_64";
      case environment::ARCH_ARM: return "ARM";
      case environment::ARCH_RISCV: return "RISC-v";
      case environment::ARCH_OTHER: return "Unknown";
   }
   return "error";
}

std::ostream& operator<<(std::ostream& os, const chainbase::environment& dt) {
   os << std::right << std::setw(17) << "Compiler: " << dt.compiler << std::endl;
   os << std::right << std::setw(17) << "Debug: " << (dt.debug ? "Yes" : "No") << std::endl;
   os << std::right << std::setw(17) << "OS: " << print_os(dt.os) << std::endl;
   os << std::right << std::setw(17) << "Arch: " << print_arch(dt.arch) << std::endl;
   os << std::right << std::setw(17) << "Boost: " << dt.boost_version/100000 << "."
                                                  << dt.boost_version/100%1000 << "."
                                                  << dt.boost_version%100 << std::endl;
   return os;
}

}
