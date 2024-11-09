use std::fmt::{Debug, Formatter, Result};
use std::mem::ManuallyDrop;
use std::ptr;

use crate::expr::buffer::Buffer;
use crate::expr::expression::Expression;
use crate::expr::iter::Iter;
use crate::slice::Slice;

/// Expression that moves elements out of an array.
pub struct IntoExpr<B: Buffer> {
    buffer: B,
    index: usize,
}

impl<B: Buffer> IntoExpr<B> {
    pub(crate) fn new(buffer: B) -> Self {
        Self { buffer, index: 0 }
    }
}

impl<B: Buffer> AsMut<Slice<B::Item, B::Shape>> for IntoExpr<B> {
    fn as_mut(&mut self) -> &mut Slice<B::Item, B::Shape> {
        debug_assert!(self.index == 0, "expression in use");

        unsafe {
            &mut *(self.buffer.as_mut_slice() as *mut Slice<ManuallyDrop<B::Item>, B::Shape>
                as *mut Slice<B::Item, B::Shape>)
        }
    }
}

impl<B: Buffer> AsRef<Slice<B::Item, B::Shape>> for IntoExpr<B> {
    fn as_ref(&self) -> &Slice<B::Item, B::Shape> {
        debug_assert!(self.index == 0, "expression in use");

        unsafe {
            &*(self.buffer.as_slice() as *const Slice<ManuallyDrop<B::Item>, B::Shape>
                as *const Slice<B::Item, B::Shape>)
        }
    }
}

impl<B: Buffer + Clone> Clone for IntoExpr<B> {
    fn clone(&self) -> Self {
        assert!(self.index == 0, "expression in use");

        Self { buffer: self.buffer.clone(), index: 0 }
    }

    fn clone_from(&mut self, source: &Self) {
        assert!(self.index == 0 && source.index == 0, "expression in use");

        self.buffer.clone_from(&source.buffer);
    }
}

impl<B: Buffer<Item: Debug>> Debug for IntoExpr<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("IntoExpr").field(&self.as_ref()).finish()
    }
}

impl<B: Buffer + Default> Default for IntoExpr<B> {
    fn default() -> Self {
        Self { buffer: Default::default(), index: 0 }
    }
}

impl<B: Buffer> Drop for IntoExpr<B> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.buffer.as_mut_slice().as_mut_ptr().add(self.index) as *mut B::Item;
            let len = self.buffer.as_slice().len() - self.index;

            ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

impl<B: Buffer> Expression for IntoExpr<B> {
    type Shape = B::Shape;

    const IS_REPEATABLE: bool = false;

    fn shape(&self) -> &Self::Shape {
        self.buffer.as_slice().shape()
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> B::Item {
        debug_assert!(self.index < self.buffer.as_slice().len(), "index out of bounds");

        self.index += 1; // Keep track of that the element is moved out.

        ManuallyDrop::take(&mut *self.buffer.as_mut_slice().as_mut_ptr().add(self.index - 1))
    }

    fn inner_rank(&self) -> usize {
        usize::MAX
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<B: Buffer> IntoIterator for IntoExpr<B> {
    type Item = B::Item;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}
