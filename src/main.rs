use nanograd::engine::Value;
use nanograd::nn::NeuralNetwork;
fn main() {
    println!("Hello from nanograd, I'm a scalar value autograd engine written in rust!");
    let a = Value::new(5.0);
    println!("The value here is: {}", a);
    let c = a + Value::new(7.0);
    let e = c * Value::new(10.0);
    println!("The new value is: {}", e);
}
