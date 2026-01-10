//! Module for array buffer handling.

#[cfg(feature = "nightly")]
use alloc::alloc::Allocator;

use core::mem::MaybeUninit;

#[cfg(not(feature = "nightly"))]
use crate::allocator::Allocator;
use crate::dim::Const;
use crate::expr::Expression;
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;

mod drain;
mod dyn_buffer;
mod static_buffer;

pub use drain::Drain;
pub use dyn_buffer::DynBuffer;
pub use static_buffer::StaticBuffer;

/// Array buffer trait.
pub trait Buffer {
    /// Array element type.
    type Item;

    /// Array shape type.
    type Shape: Shape;

    /// Returns a mutable slice containing the array buffer.
    fn as_mut_slice(&mut self) -> &mut Slice<Self::Item, Self::Shape>;

    /// Returns a slice containing the array buffer.
    fn as_slice(&self) -> &Slice<Self::Item, Self::Shape>;
}

/// Buffer trait for owned arrays.
pub trait Owned: Buffer {
    /// Allocator type.
    type Alloc: Allocator;

    #[doc(hidden)]
    type WithConst<const N: usize>: Owned<
            Item = Self::Item,
            Shape = <Self::Shape as Shape>::Prepend<Const<N>>,
            Alloc = Self::Alloc,
        >;

    #[doc(hidden)]
    fn allocator(&self) -> &Self::Alloc;

    #[doc(hidden)]
    unsafe fn cast<T>(self) -> <Self::Shape as Shape>::Buffer<T, Self::Alloc>;

    #[doc(hidden)]
    fn clone(&self) -> Self
    where
        Self::Item: Clone,
        Self::Alloc: Clone;

    #[doc(hidden)]
    fn clone_from(&mut self, source: &Self)
    where
        Self::Item: Clone,
        Self::Alloc: Clone;

    #[doc(hidden)]
    fn clone_from_slice(&mut self, slice: &Slice<Self::Item, Self::Shape>)
    where
        Self::Item: Clone;

    #[doc(hidden)]
    fn from_dyn(buffer: DynBuffer<Self::Item, Self::Shape, Self::Alloc>) -> Self;

    #[doc(hidden)]
    fn from_static<S: ConstShape>(
        buffer: StaticBuffer<Self::Item, S, Self::Alloc>,
        new_shape: Self::Shape,
    ) -> Self;

    #[doc(hidden)]
    fn into_buffer<B: Owned<Item = Self::Item, Alloc = Self::Alloc>>(self) -> B;

    #[doc(hidden)]
    fn into_dyn(self) -> DynBuffer<Self::Item, Self::Shape, Self::Alloc>;

    #[doc(hidden)]
    fn into_shape<S: Shape>(self, new_shape: S) -> S::Buffer<Self::Item, Self::Alloc>;

    #[doc(hidden)]
    fn uninit_in(
        shape: Self::Shape,
        alloc: Self::Alloc,
    ) -> <Self::Shape as Shape>::Buffer<MaybeUninit<Self::Item>, Self::Alloc>;

    #[doc(hidden)]
    fn zip_with<T, E, F>(self, expr: E, f: F) -> <Self::Shape as Shape>::Buffer<T, Self::Alloc>
    where
        E: Expression,
        F: FnMut((Self::Item, E::Item)) -> T;
}
