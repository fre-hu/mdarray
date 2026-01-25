use core::fmt::{Debug, Formatter, Result};
use core::mem::ManuallyDrop;
use core::ptr;

use crate::buffer::Buffer;
use crate::expr::expression::Expression;
use crate::expr::iter::Iter;
use crate::slice::Slice;

/// Expression that moves elements out of an array.
pub struct IntoExpr<T, B: Buffer<Item = ManuallyDrop<T>>> {
    buffer: B,
    index: usize,
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> IntoExpr<T, B> {
    /// Creates an expression from an array buffer.
    #[inline]
    pub fn new(buffer: B) -> Self {
        Self { buffer, index: 0 }
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> AsMut<Slice<T, B::Shape>> for IntoExpr<T, B> {
    #[inline]
    fn as_mut(&mut self) -> &mut Slice<T, B::Shape> {
        debug_assert!(self.index == 0, "expression in use");

        unsafe {
            &mut *(self.buffer.as_mut_slice() as *mut Slice<ManuallyDrop<T>, B::Shape>
                as *mut Slice<T, B::Shape>)
        }
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> AsRef<Slice<T, B::Shape>> for IntoExpr<T, B> {
    #[inline]
    fn as_ref(&self) -> &Slice<T, B::Shape> {
        debug_assert!(self.index == 0, "expression in use");

        unsafe {
            &*(self.buffer.as_slice() as *const Slice<ManuallyDrop<T>, B::Shape>
                as *const Slice<T, B::Shape>)
        }
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>> + Clone> Clone for IntoExpr<T, B> {
    #[inline]
    fn clone(&self) -> Self {
        assert!(self.index == 0, "expression in use");

        Self { buffer: self.buffer.clone(), index: 0 }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        assert!(self.index == 0 && source.index == 0, "expression in use");

        self.buffer.clone_from(&source.buffer);
    }
}

impl<T: Debug, B: Buffer<Item = ManuallyDrop<T>>> Debug for IntoExpr<T, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("IntoExpr").field(&self.as_ref()).finish()
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>> + Default> Default for IntoExpr<T, B> {
    #[inline]
    fn default() -> Self {
        Self { buffer: Default::default(), index: 0 }
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> Drop for IntoExpr<T, B> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let ptr = self.buffer.as_mut_slice().as_mut_ptr().add(self.index) as *mut T;
            let len = self.buffer.as_slice().len() - self.index;

            ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> Expression for IntoExpr<T, B> {
    type Shape = B::Shape;

    const IS_REPEATABLE: bool = false;

    #[inline]
    fn shape(&self) -> &B::Shape {
        self.buffer.as_slice().shape()
    }

    #[inline]
    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        debug_assert!(self.index < self.buffer.as_slice().len(), "index out of bounds");

        self.index += 1; // Keep track of that the element is moved out.

        unsafe {
            ManuallyDrop::take(&mut *self.buffer.as_mut_slice().as_mut_ptr().add(self.index - 1))
        }
    }

    #[inline]
    fn inner_rank(&self) -> usize {
        usize::MAX
    }

    #[inline]
    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}

    #[inline]
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<T, B: Buffer<Item = ManuallyDrop<T>>> IntoIterator for IntoExpr<T, B> {
    type Item = T;
    type IntoIter = Iter<Self>;

    #[inline]
    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}
