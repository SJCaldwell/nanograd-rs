use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul};
use std::rc::Rc;

pub struct _Value {
    pub data: f32,
    pub _prev: Vec<Rc<RefCell<_Value>>>,
}

pub struct Value(Rc<RefCell<_Value>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(_Value {
            data,
            _prev: vec![],
        })))
    }

    pub fn new_with_parents(data: f32, parents: Vec<Value>) -> Self {
        let parent_refs = parents.into_iter().map(|p| p.0.clone()).collect();
        Value(Rc::new(RefCell::new(_Value {
            data,
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
        Value::new_with_parents(new_data, vec![self, other])
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        let new_data = self.0.borrow().data * other.0.borrow().data;
        Value::new_with_parents(new_data, vec![self, other])
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value_borrow = self.0.borrow();
        write!(f, "Value(data={})", value_borrow.data)
    }
}
