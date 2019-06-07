use influx_db_client as influxdb;
use crate::metrics;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use buffett_timing::timing;

const DEFAULT_METRICS_RATE: usize = 100;

pub struct Counter {
    pub name: &'static str,
    pub counts: AtomicUsize,
    pub times: AtomicUsize,
    pub lastlog: AtomicUsize,
    pub lograte: AtomicUsize,
}

#[macro_export]
macro_rules! new_counter {
    ($name:expr, $lograte:expr) => {
        Counter {
            name: $name,
            counts: AtomicUsize::new(0),
            times: AtomicUsize::new(0),
            lastlog: AtomicUsize::new(0),
            lograte: AtomicUsize::new($lograte),
        }
    };
}

#[macro_export]
macro_rules! sub_counter {
    ($name:expr, $count:expr) => {
        unsafe { $name.inc($count) };
    };
}

#[macro_export]
macro_rules! sub_new_counter_info {
    ($name:expr, $count:expr) => {{
        sub_new_counter!($name, $count, Level::Info, 0);
    }};
    ($name:expr, $count:expr, $lograte:expr) => {{
        sub_new_counter!($name, $count, Level::Info, $lograte);
    }};
}

#[macro_export]
macro_rules! sub_new_counter {($name:expr, $count:expr, $level:expr, $lograte:expr) => 
        {{if log_enabled!($level) {static mut INC_NEW_COUNTER: Counter = new_counter!($name, $lograte);
            sub_counter!(INC_NEW_COUNTER, $count);
        }
    }};
}

impl Counter {
    fn default_log_rate() -> usize {
        let v = env::var("BITCONCH_DASHBOARD_RATE")
            .map(|x| x.parse().unwrap_or(DEFAULT_METRICS_RATE))
            .unwrap_or(DEFAULT_METRICS_RATE);
        if v == 0 {
            DEFAULT_METRICS_RATE
        } else {
            v
        }
    }
    pub fn inc(&mut self, events: usize) {
        let counts = self.counts.fetch_add(events, Ordering::Relaxed);
        let times = self.times.fetch_add(1, Ordering::Relaxed);
        let mut lograte = self.lograte.load(Ordering::Relaxed);
        if lograte == 0 {
            lograte = Counter::default_log_rate();
            self.lograte.store(lograte, Ordering::Relaxed);
        }
        if times % lograte == 0 && times > 0 {
            let lastlog = self.lastlog.load(Ordering::Relaxed);
            info!(
                "COUNTER:{{\"name\": \"{}\", \"counts\": {}, \"samples\": {},  \"now\": {}, \"events\": {}}}",
                self.name,
                counts + events,
                times,
                timing::timestamp(),
                events,
            );
            metrics::submit(
                influxdb::Point::new(&format!("counter-{}", self.name))
                    .add_field(
                        "count",
                        influxdb::Value::Integer(counts as i64 - lastlog as i64),
                    ).to_owned(),
            );
            self.lastlog
                .compare_and_swap(lastlog, counts, Ordering::Relaxed);
        }
    }
}
