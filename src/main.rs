use nanograd::engine::Value;
fn main() {
    let input = Value::new(2.0);
    let output = input.relu();
    output.backward();
}
