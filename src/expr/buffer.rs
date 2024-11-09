#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::mem::ManuallyDrop;
use std::ptr;

#[cfg(not(feature = "nightly"))]
use crate::alloc::{Allocator, Global};
use crate::array::Array;
use crate::index::{Axis, Nth};
use crate::mapping::Mapping;
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;
use crate::tensor::Tensor;
use crate::view::ViewMut;

/// Array buffer trait, for moving elements out of an array.
pub trait Buffer {
    /// Array element type.
    type Item;

    /// Array shape type.
    type Shape: Shape;

    #[doc(hidden)]
    fn as_mut_slice(&mut self) -> &mut Slice<ManuallyDrop<Self::Item>, Self::Shape>;

    #[doc(hidden)]
    fn as_slice(&self) -> &Slice<ManuallyDrop<Self::Item>, Self::Shape>;
}

/// Buffer for moving elements out of an array range.
pub struct Drain<'a, T, S: Shape, A: Allocator = Global> {
    tensor: &'a mut Tensor<T, S, A>,
    view: ViewMut<'a, ManuallyDrop<T>, S>,
    new_size: usize,
    tail: usize,
}

impl<'a, T, S: Shape, A: Allocator> Drain<'a, T, S, A> {
    pub(crate) fn new(tensor: &'a mut Tensor<T, S, A>, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= tensor.dim(0), "invalid range");

        let new_size = tensor.dim(0) - (end - start);
        let tail = <Nth<0> as Axis>::resize(tensor.mapping(), new_size - start).len();

        // Shrink the array, to be safe in case Drain is leaked.
        unsafe {
            tensor.set_mapping(Mapping::resize_dim(tensor.mapping(), 0, start));
        }

        let ptr = unsafe { tensor.as_mut_ptr().add(tensor.len()) as *mut ManuallyDrop<T> };
        let mapping = Mapping::resize_dim(tensor.mapping(), 0, end - start);

        let view = unsafe { ViewMut::new_unchecked(ptr, mapping) };

        Self { tensor, view, new_size, tail }
    }
}

impl<T, S: Shape, A: Allocator> Buffer for Drain<'_, T, S, A> {
    type Item = T;
    type Shape = S;

    fn as_mut_slice(&mut self) -> &mut Slice<ManuallyDrop<T>, S> {
        &mut self.view
    }

    fn as_slice(&self) -> &Slice<ManuallyDrop<T>, S> {
        &self.view
    }
}

impl<T, S: Shape, A: Allocator> Drop for Drain<'_, T, S, A> {
    fn drop(&mut self) {
        let mapping = Mapping::resize_dim(self.tensor.mapping(), 0, self.new_size);

        unsafe {
            ptr::copy(self.view.as_ptr().add(self.view.len()), self.view.as_mut_ptr(), self.tail);
            self.tensor.set_mapping(mapping);
        }
    }
}

impl<T, S: ConstShape> Buffer for Array<ManuallyDrop<T>, S> {
    type Item = T;
    type Shape = S;

    fn as_mut_slice(&mut self) -> &mut Slice<ManuallyDrop<T>, S> {
        self
    }

    fn as_slice(&self) -> &Slice<ManuallyDrop<T>, S> {
        self
    }
}

impl<T, S: Shape, A: Allocator> Buffer for Tensor<ManuallyDrop<T>, S, A> {
    type Item = T;
    type Shape = S;

    fn as_mut_slice(&mut self) -> &mut Slice<ManuallyDrop<T>, S> {
        self
    }

    fn as_slice(&self) -> &Slice<ManuallyDrop<T>, S> {
        self
    }
}
