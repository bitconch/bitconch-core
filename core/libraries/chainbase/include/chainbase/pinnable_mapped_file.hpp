#pragma once

#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/filesystem.hpp>
#include <boost/asio/io_service.hpp>

namespace chainbase {

namespace bip = boost::interprocess;
namespace bfs = boost::filesystem;

class pinnable_mapped_file {
   public:
      typedef typename bip::managed_mapped_file::segment_manager segment_manager;

      enum map_mode {
         mapped,
         heap,
         locked
      };

      pinnable_mapped_file(const bfs::path& dir, bool writable, uint64_t shared_file_size, bool allow_dirty, map_mode mode, std::vector<std::string> hugepage_paths);
      pinnable_mapped_file(pinnable_mapped_file&& o);
      pinnable_mapped_file& operator=(pinnable_mapped_file&&);
      pinnable_mapped_file(const pinnable_mapped_file&) = delete;
      pinnable_mapped_file& operator=(const pinnable_mapped_file&) = delete;
      ~pinnable_mapped_file();

      segment_manager* get_segment_manager() const { return _segment_manager;}

   private:
      void                                          set_mapped_file_db_dirty(bool);
      void                                          load_database_file(boost::asio::io_service& sig_ios);
      void                                          save_database_file();
      bool                                          all_zeros(char* data, size_t sz);
      bip::mapped_region                            get_huge_region(const std::vector<std::string>& huge_paths);

      bip::file_lock                                _mapped_file_lock;
      bfs::path                                     _data_file_path;
      std::string                                   _database_name;
      bool                                          _writable;

      bip::mapped_region                            _file_mapped_region;
      bip::mapped_region                            _mapped_region;

#ifdef _WIN32
      bip::permissions                              _db_permissions;
#else
      bip::permissions                              _db_permissions{S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH};
#endif

      segment_manager*                              _segment_manager = nullptr;

      constexpr static unsigned                     _db_size_multiple_requirement = 1024*1024; //1MB
};

std::istream& operator>>(std::istream& in, pinnable_mapped_file::map_mode& runtime);
std::ostream& operator<<(std::ostream& osm, pinnable_mapped_file::map_mode m);

}
