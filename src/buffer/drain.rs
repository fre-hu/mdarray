#[cfg(feature = "nightly")]
use alloc::alloc::{Allocator, Global};

use core::mem::ManuallyDrop;
use core::ptr;

#[cfg(not(feature = "nightly"))]
use crate::allocator::{Allocator, Global};
use crate::array::Array;
use crate::buffer::{Buffer, DynBuffer};
use crate::dim::Const;
use crate::index::Axis;
use crate::mapping::Mapping;
use crate::shape::Shape;
use crate::slice::Slice;
use crate::view::ViewMut;

/// Buffer for moving elements out of an array range.
pub struct Drain<'a, T, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator = Global> {
    array: &'a mut Array<T, S, A>,
    view: ViewMut<'a, ManuallyDrop<T>, S>,
    new_size: usize,
    tail: usize,
}

impl<'a, T, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator> Drain<'a, T, S, A> {
    #[inline]
    pub(crate) fn new(array: &'a mut Array<T, S, A>, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= array.dim(0), "invalid range");

        let new_size = array.dim(0) - (end - start);
        let tail = Axis::resize(Const::<0>, array.mapping(), new_size - start).len();

        // Shrink the array, to be safe in case Drain is leaked.
        unsafe {
            array.set_shape(Shape::resize_dim(array.shape(), 0, start));
        }

        let ptr = unsafe { array.as_mut_ptr().add(array.len()) as *mut ManuallyDrop<T> };
        let mapping = Mapping::resize_dim(array.mapping(), 0, end - start);

        let view = unsafe { ViewMut::new_unchecked(ptr, mapping) };

        Self { array, view, new_size, tail }
    }
}

impl<T, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator> Buffer for Drain<'_, T, S, A> {
    type Item = ManuallyDrop<T>;
    type Shape = S;

    #[inline]
    fn as_mut_slice(&mut self) -> &mut Slice<ManuallyDrop<T>, S> {
        &mut self.view
    }

    #[inline]
    fn as_slice(&self) -> &Slice<ManuallyDrop<T>, S> {
        &self.view
    }
}

impl<T, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator> Drop for Drain<'_, T, S, A> {
    #[inline]
    fn drop(&mut self) {
        let shape = Shape::resize_dim(self.array.shape(), 0, self.new_size);

        unsafe {
            ptr::copy(self.view.as_ptr().add(self.view.len()), self.view.as_mut_ptr(), self.tail);
            self.array.set_shape(shape);
        }
    }
}
