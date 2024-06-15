use std::cell::RefCell;
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
}

pub struct _Value {
    pub data: f32,
    pub grad: f32, //gradient used during the background pass
    pub _prev: Vec<Rc<RefCell<_Value>>>,
    pub op: Op, //Operation that created the value
}

pub struct Value(Rc<RefCell<_Value>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(_Value {
            data,
            grad: 0.0,
            op: Op::Init,
            _prev: vec![],
        })))
    }

    pub fn new_with_parents(data: f32, parents: Vec<Value>, op: Op) -> Self {
        let parent_refs = parents.into_iter().map(|p| p.0.clone()).collect();
        Value(Rc::new(RefCell::new(_Value {
            data,
            grad: 0.0,
            op,
            _prev: parent_refs,
        })))
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
