use influx_db_client as influxdb;
use std::env;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Barrier, Mutex, Once, ONCE_INIT};
use std::thread;
use std::time::{Duration, Instant};
use sys_info::hostname;
use buffett_timing::timing;

#[derive(Debug)]
/// define enueration of MetricsComman with the variants of Submit and Flush
enum MetricsCommand {
    Submit(influxdb::Point),
    Flush(Arc<Barrier>),
}

/// define struct of MetricsAgent with the fields of sender
struct MetricsAgent {
    sender: Sender<MetricsCommand>,
}

/// MetricsWriter trait that consists of the behavior provided by a "write" method
trait MetricsWriter {
    
    fn write(&self, points: Vec<influxdb::Point>);
}

/// define struct of DbWriter with the fields of client
struct DbWriter {
    client: Option<influxdb::Client>,
}

/// implementing new  methods on DbWriter structure
impl DbWriter {
    /// define the function of new, and return a instance of DbWriter
    fn new() -> Self {
        DbWriter {
            client: Self::build_client(),
        }
    }

    /// define the function of "build_client",
    /// its return value type is Option <T> where T is influxdb:: Client
    fn build_client() -> Option<influxdb::Client> {
        /// fetches the environment variable key of "INFLUX_HOST" from the current process 
        /// if failed then return "https://dashboard.bitconch.org"
        let host = env::var("INFLUX_HOST")
            .unwrap_or_else(|_| "https://dashboard.bitconch.org".to_string());
        /// fetches the environment variable key of "IINFLUX_DATABASE" from the current process 
        /// if failed then return "scratch"
        let db = env::var("INFLUX_DATABASE").unwrap_or_else(|_| "scratch".to_string());
        /// fetches the environment variable key of "INFLUX_USERNAME" from the current process 
        /// if failed then return "scratch_writer"
        let username = env::var("INFLUX_USERNAME").unwrap_or_else(|_| "scratch_writer".to_string());
        /// fetches the environment variable key of "NFLUX_PASSWORD" from the current process 
        /// if failed then return "topsecret"
        let password = env::var("INFLUX_PASSWORD").unwrap_or_else(|_| "topsecret".to_string());

        /// logs the message at the debug level
        debug!("InfluxDB host={} db={} username={}", host, db, username);
        /// create a new influxdb client with https of "host, db, None"
        /// and change the client's user
        let mut client = influxdb::Client::new_with_option(host, db, None)
            .set_authentication(username, password);

        /// set the read timeout value, unit "s" 
        client.set_read_timeout(1 /*second*/);
        /// set the write timeout value, unit "s" 
        client.set_write_timeout(1 /*second*/);

        /// logs the message of the version of the InfluxDB database at the debug level
        debug!("InfluxDB version: {:?}", client.get_version());
        ///  "client" is wrapped in "Some" and returned
        Some(client)
    }
}

/// define write function to implement MetricsWriter trait on DbWriter structure
impl MetricsWriter for DbWriter {
    fn write(&self, points: Vec<influxdb::Point>) {
        /// destructure the fields "client" of DbWriter in "Some" value
        /// logs the length of "points" at the debug level 
        if let Some(ref client) = self.client {
            debug!("submitting {} points", points.len());
            /// write multiple points to the influxdb database
            /// if failed then format print the error message
            if let Err(err) = client.write_points(
                influxdb::Points { point: points },
                Some(influxdb::Precision::Milliseconds),
                None,
            ) {
                debug!("DbWriter write error: {:?}", err);
            }
        }
    }
}

/// define default function to implement Default trait on MetricsAgent structure
/// and return a instance of MetricsAgent
impl Default for MetricsAgent {
    fn default() -> Self {
        Self::new(
            Arc::new(DbWriter::new()),
            Duration::from_secs(10),
        )
    }
}

/// implementing new and run methods on MetricsAgent structure
impl MetricsAgent {
    /// define the function of new, and return a instance of MetricsAgent
    fn new(metrics_writer: Arc<MetricsWriter + Send + Sync>, write_frequency: Duration) -> Self {
        /// creates a new asynchronous channel, 
        /// returning the sender/receiver halves, the type of  channel is MetricsCommand
        let (sender, receiver) = channel::<MetricsCommand>();
        /// spawns a new thread to start the run function
        thread::spawn(move || Self::run(&receiver, &metrics_writer, write_frequency));
        /// eturn the instance of MetricsAgent
        MetricsAgent { sender }
    }

    /// define the function of run
    fn run(
        receiver: &Receiver<MetricsCommand>,
        writer: &Arc<MetricsWriter + Send + Sync>,
        write_frequency: Duration,
    ) {
        /// logs the message at the trace leve
        trace!("run: enter");
        /// get the current time
        let mut last_write_time = Instant::now();
        /// constructs a empty vector
        let mut points = Vec::new();

        /// start loop
        loop {
            /// destructure the value on receiver with the parameter of "write_frequency / 2"
            match receiver.recv_timeout(write_frequency / 2) {
                /// if is a "OK" value, then destructure MetricsCommand enum
                Ok(cmd) => match cmd {
                    /// if the variants of MetricsCommand enum is "Flush(barrier)"
                    /// then logs the message at the debug level
                    MetricsCommand::Flush(barrier) => {
                        debug!("metrics_thread: flush");
                        /// if the vector of "points" is not empty
                        /// then write the "points" vector into influxdb database
                        /// constructs a empty vector
                        /// get the current time to write into the influxdb database
                        if !points.is_empty() {
                            writer.write(points);
                            points = Vec::new();
                            last_write_time = Instant::now();
                        }
                        /// blocks the current thread of "barrier" until all threads have rendezvoused here.
                        barrier.wait();
                    }
                    /// if the variants of MetricsCommand enum is "Submit(point)"
                    /// then logs the message at the debug level
                    /// and appends the element of "point" to the back of "points" vector
                    MetricsCommand::Submit(point) => {
                        debug!("run: submit {:?}", point);
                        points.push(point);
                    }
                },
                /// if is a "Err" value of "Timeout", then logs the message at the trace leve
                Err(RecvTimeoutError::Timeout) => {
                    trace!("run: receive timeout");
                }
                /// if is a "Err" value of "Disconnected",
                /// then logs the message at the debug level, and quit the loop
                Err(RecvTimeoutError::Disconnected) => {
                    debug!("run: sender disconnected");
                    break;
                }
            }

            let now = Instant::now();
            if now.duration_since(last_write_time) >= write_frequency && !points.is_empty() {
                debug!("run: writing {} points", points.len());
                writer.write(points);
                points = Vec::new();
                last_write_time = now;
            }
        }
        trace!("run: exit");
    }

    /// declare the function of submit
    pub fn submit(&self, mut point: influxdb::Point) {
        /// if the timestamp of the point is none
        /// then create timestamp in timing
        if point.timestamp.is_none() {
            point.timestamp = Some(timing::timestamp() as i64);
        }
        debug!("Submitting point: {:?}", point);
        /// send the value of "Submit(point)" on sender channel
        self.sender.send(MetricsCommand::Submit(point)).unwrap();
    }

    /// declare the function of flush
    pub fn flush(&self) {
        /// logs the message at the debug level
        debug!("Flush");
        /// creates barrier that can block 2 threads
        let barrier = Arc::new(Barrier::new(2));
        /// send the value of "Flush(Arc::clone(&barrier))" on sender channel
        self.sender
            .send(MetricsCommand::Flush(Arc::clone(&barrier)))
            .unwrap();

        /// blocks the current thread of "barrier" until all threads have rendezvoused here
        barrier.wait();
    }
}

/// define drop method to implement Drop trait on MetricsAgent structure
/// disposes the value of flush function
impl Drop for MetricsAgent {
    fn drop(&mut self) {
        self.flush();
    }
}

/// declare the function of flush get_mutex_agent
/// with the return value type is MetricsAgent structure
fn get_mutex_agent() -> Arc<Mutex<MetricsAgent>> {
    /// declare a static named "INIT" with the type of synchronization primitive
    /// and initialization value for static `Once` values
    static INIT: Once = ONCE_INIT;
    /// declare a mutable static named "AGENT" with the type of Option<Arc<Mutex<MetricsAgent>>>
    /// and initialization to None
    static mut AGENT: Option<Arc<Mutex<MetricsAgent>>> = None;
    unsafe {
        INIT.call_once(|| AGENT = Some(Arc::new(Mutex::new(MetricsAgent::default()))));
        match AGENT {
            Some(ref agent) => agent.clone(),
            None => panic!("Failed to initialize metrics agent"),
        }
    }
}


pub fn submit(point: influxdb::Point) {
    let agent_mutex = get_mutex_agent();
    let agent = agent_mutex.lock().unwrap();
    agent.submit(point);
}


pub fn flush() {
    let agent_mutex = get_mutex_agent();
    let agent = agent_mutex.lock().unwrap();
    agent.flush();
}


pub fn set_panic_hook(program: &'static str) {
    use std::panic;
    static SET_HOOK: Once = ONCE_INIT;
    SET_HOOK.call_once(|| {
        let default_hook = panic::take_hook();
        panic::set_hook(Box::new(move |ono| {
            default_hook(ono);
            submit(
                influxdb::Point::new("panic")
                    .add_tag("program", influxdb::Value::String(program.to_string()))
                    .add_tag(
                        "thread",
                        influxdb::Value::String(
                            thread::current().name().unwrap_or("?").to_string(),
                        ),
                    )
                    
                    .add_field("one", influxdb::Value::Integer(1))
                    .add_field(
                        "message",
                        influxdb::Value::String(
                            
                            ono.to_string(),
                        ),
                    )
                    .add_field(
                        "location",
                        influxdb::Value::String(match ono.location() {
                            Some(location) => location.to_string(),
                            None => "?".to_string(),
                        }),
                    )
                    .add_field(
                        "host",
                        influxdb::Value::String(
                            hostname().unwrap_or_else(|_| "?".to_string())
                        ),
                    )
                    .to_owned(),
            );
            
            flush();
        }));
    });
}

