use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

#[derive(Debug)]
pub enum Op {
    Init, //Indicates no operation was used to make it
    Add,
    Mul,
    Div,
    Sub,
    ReLU,
}

pub struct _Value {
    pub data: f32,
    pub grad: f32, //gradient used during the background pass
    pub _prev: Vec<Rc<RefCell<_Value>>>,
    pub op: Op,                          //Operation that created the value
    _backward: Option<Box<dyn FnMut()>>, //Closure to compute the gradient
}

pub struct Value(Rc<RefCell<_Value>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(_Value {
            data,
            grad: 0.0,
            op: Op::Init,
            _prev: vec![],
            _backward: None,
        })))
    }

    pub fn new_with_parents(data: f32, parents: Vec<Value>, op: Op) -> Self {
        let parent_refs = parents.into_iter().map(|p| p.0.clone()).collect();
        Value(Rc::new(RefCell::new(_Value {
            data,
            grad: 0.0,
            op,
            _prev: parent_refs,
            _backward: None,
        })))
    }

    pub fn backward(&self) {
        let mut topo: Vec<Rc<RefCell<_Value>>> = vec![];
        let mut visited: HashSet<*const _Value> = HashSet::new();
        let mut stack: Vec<Rc<RefCell<_Value>>> = vec![self.0.clone()];

        while let Some(node) = stack.pop() {
            let node_ptr = Rc::as_ptr(&node) as *const _Value;
            if !visited.contains(&node_ptr) {
                visited.insert(node_ptr);
                topo.push(node.clone());
                for parent in &node.borrow()._prev {
                    stack.push(parent.clone());
                }
            }
        }
        //Reverse the topological order to start at the root node and go backwards
        topo.reverse();

        //Initialize the gradient of the output node
        if let Some(output_node) = topo.first() {
            output_node.borrow_mut().grad = 1.0;
        }

        for node in topo {
            if let Some(ref mut backward_fn) = node.borrow_mut()._backward.take() {
                backward_fn();
            }
        }
    }

    pub fn relu(&self) -> Value {
        let data = self.0.borrow().data;
        let new_data = data.max(0.0); // ReLU function
        let new_value = Value::new_with_parents(new_data, vec![self.clone()], Op::ReLU);

        // Create a weak reference to new_value and self before defining the closure
        let new_value_weak = Rc::downgrade(&new_value.0);
        let self_weak = Rc::downgrade(&self.0);

        // Set up the backward function using only weak references
        new_value.0.borrow_mut()._backward = Some(Box::new(move || {
            if let (Some(self_rc), Some(new_value_rc)) =
                (self_weak.upgrade(), new_value_weak.upgrade())
            {
                let mut self_borrow = self_rc.borrow_mut();
                let new_value_grad = new_value_rc.borrow().grad;
                self_borrow.grad += if data > 0.0 { new_value_grad } else { 0.0 };
            }
        }));

        new_value
    }

    pub fn display_parents(&self) -> String {
        let value_borrow = self.0.borrow();
        let parents = &value_borrow._prev;

        //Build a string containing all parent values
        let parent_strings: Vec<String> = parents
            .iter()
            .map(|parent| {
                let parent_borrow = parent.borrow();
                format!("Value(data={})", parent_borrow.data)
            })
            .collect();

        //Join strings
        parent_strings.join(", ")
    }
}
impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let new_data = self.0.borrow().data + other.0.borrow().data;
        Value::new_with_parents(new_data, vec![self, other], Op::Add)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        let new_data = self.0.borrow().data * other.0.borrow().data;
        Value::new_with_parents(new_data, vec![self, other], Op::Mul)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        let new_data = self.0.borrow().data - other.0.borrow().data;
        Value::new_with_parents(new_data, vec![self, other], Op::Sub)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        let new_data = self.0.borrow().data / other.0.borrow().data;
        Value::new_with_parents(new_data, vec![self, other], Op::Div)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value_borrow = self.0.borrow();
        write!(
            f,
            "Value(data={}, grad={}, op={:?})",
            value_borrow.data, value_borrow.grad, value_borrow.op
        )
    }
}

impl Clone for Value {
    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}
