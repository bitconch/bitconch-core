#[macro_use]
extern crate influx_db_client;

use influx_db_client::{Client, Point, Points, Value, Precision};
use std::env;

fn main() {
    // default with "http://127.0.0.1:8086", db with "test"
    //let client = Client::default().set_authentication("caesar", "zaq12wsx");
    let mut client = Client::new_with_option("https://localhost:8086", "dashboard01", None)
            .set_authentication("caesar", "zaq12wsx");
    println!("step1");
    let mut point = point!("test1");
    point
        .add_field("foo", Value::String("bar".to_string()))
        .add_field("integer", Value::Integer(11))
        .add_field("float", Value::Float(22.3))
        .add_field("'boolean'", Value::Boolean(false));

    let point1 = Point::new("test1")
        .add_tag("tags", Value::String(String::from("\\\"fda")))
        .add_tag("number", Value::Integer(12))
        .add_tag("float", Value::Float(12.6))
        .add_field("fd", Value::String("'3'".to_string()))
        .add_field("quto", Value::String("\\\"fda".to_string()))
        .add_field("quto1", Value::String("\"fda".to_string()))
        .to_owned();

    let points = points!(point1, point);
    println!("step2");
    // if Precision is None, the default is second
    // Multiple write
    let _rc = client.write_points(points, Some(Precision::Seconds), None);
    println!("InfluxDbMetricsWriter write error: {:?}", _rc);
    println!("step3");
    // query, it's type is Option<Vec<Node>>
    let res = client.query("select * from test1", None).unwrap();
    println!("step4");
    println!("{:?}", res.unwrap()[0].series)
}

fn get_env_settings() -> (String, String, String, String) {
    let host =
        env::var("INFLUX_HOST").unwrap_or_else(|_| "https://localhost:8086".to_string());
    let db = env::var("INFLUX_DATABASE").unwrap_or_else(|_| "scratch".to_string());
    let username = env::var("INFLUX_USERNAME").unwrap_or_else(|_| "scratch_writer".to_string());
    let password = env::var("INFLUX_PASSWORD").unwrap_or_else(|_| "topsecret".to_string());
    (host, db, username, password)
}
