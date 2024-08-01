#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::mem::ManuallyDrop;
use std::ptr;

#[cfg(not(feature = "nightly"))]
use crate::alloc::{Allocator, Global};
use crate::array::Array;
use crate::expr::ExprMut;
use crate::grid::Grid;
use crate::index::{Axis, Outer};
use crate::mapping::Mapping;
use crate::shape::{ConstShape, Shape};
use crate::span::Span;

/// Array buffer trait, for moving elements out of an array.
pub trait Buffer {
    /// Array element type.
    type Item;

    /// Array shape type.
    type Shape: Shape;

    #[doc(hidden)]
    fn as_mut_span(&mut self) -> &mut Span<ManuallyDrop<Self::Item>, Self::Shape>;

    #[doc(hidden)]
    fn as_span(&self) -> &Span<ManuallyDrop<Self::Item>, Self::Shape>;
}

/// Buffer for moving elements out of an array range.
pub struct Drain<'a, T, S: Shape, A: Allocator = Global> {
    grid: &'a mut Grid<T, S, A>,
    view: ExprMut<'a, ManuallyDrop<T>, S>,
    new_size: usize,
    tail: usize,
}

impl<'a, T, S: Shape, A: Allocator> Drain<'a, T, S, A> {
    pub(crate) fn new(grid: &'a mut Grid<T, S, A>, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= grid.dim(S::RANK - 1), "invalid range");

        let new_size = grid.dim(S::RANK - 1) - (end - start);
        let tail = <Outer as Axis>::resize(grid.mapping(), new_size - start).len();

        // Shrink the array, to be safe in case Drain is leaked.
        unsafe {
            grid.set_mapping(Mapping::resize_dim(grid.mapping(), S::RANK - 1, start));
        }

        let ptr = unsafe { grid.as_mut_ptr().add(grid.len()) as *mut ManuallyDrop<T> };
        let mapping = Mapping::resize_dim(grid.mapping(), S::RANK - 1, end - start);

        let view = unsafe { ExprMut::new_unchecked(ptr, mapping) };

        Self { grid, view, new_size, tail }
    }
}

impl<T, S: Shape, A: Allocator> Buffer for Drain<'_, T, S, A> {
    type Item = T;
    type Shape = S;

    fn as_mut_span(&mut self) -> &mut Span<ManuallyDrop<T>, S> {
        &mut self.view
    }

    fn as_span(&self) -> &Span<ManuallyDrop<T>, S> {
        &self.view
    }
}

impl<T, S: Shape, A: Allocator> Drop for Drain<'_, T, S, A> {
    fn drop(&mut self) {
        let mapping = Mapping::resize_dim(self.grid.mapping(), S::RANK - 1, self.new_size);

        unsafe {
            ptr::copy(self.view.as_ptr().add(self.view.len()), self.view.as_mut_ptr(), self.tail);
            self.grid.set_mapping(mapping);
        }
    }
}

impl<T, S: ConstShape> Buffer for Array<ManuallyDrop<T>, S> {
    type Item = T;
    type Shape = S;

    fn as_mut_span(&mut self) -> &mut Span<ManuallyDrop<T>, S> {
        self
    }

    fn as_span(&self) -> &Span<ManuallyDrop<T>, S> {
        self
    }
}

impl<T, S: Shape, A: Allocator> Buffer for Grid<ManuallyDrop<T>, S, A> {
    type Item = T;
    type Shape = S;

    fn as_mut_span(&mut self) -> &mut Span<ManuallyDrop<T>, S> {
        self
    }

    fn as_span(&self) -> &Span<ManuallyDrop<T>, S> {
        self
    }
}
