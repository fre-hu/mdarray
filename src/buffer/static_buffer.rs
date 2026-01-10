#[cfg(all(feature = "nightly", not(feature = "std")))]
use alloc::vec::Vec;

#[cfg(not(feature = "nightly"))]
use alloc::alloc::Layout;
#[cfg(feature = "nightly")]
use alloc::alloc::{Allocator, Global, Layout};

use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

#[cfg(not(feature = "nightly"))]
use crate::allocator::{Allocator, Global};
use crate::array::Array;
use crate::buffer::dyn_buffer::DynBuffer;
use crate::buffer::{Buffer, Owned};
use crate::expr::{Expression, IntoExpr};
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;

/// Array buffer type with inline allocation.
#[repr(C)]
pub struct StaticBuffer<T, S: ConstShape, A: Allocator = Global> {
    inner: S::Inner<T>,
    alloc: A,
}

impl<T, S: ConstShape, A: Allocator> StaticBuffer<T, S, A> {
    #[inline]
    pub(crate) fn into_parts(self) -> (S::Inner<T>, A) {
        (self.inner, self.alloc)
    }
}

impl<T, S: ConstShape, A: Allocator> Buffer for StaticBuffer<T, S, A> {
    type Item = T;
    type Shape = S;

    #[inline]
    fn as_mut_slice(&mut self) -> &mut Slice<T, S> {
        unsafe { &mut *(&mut self.inner as *mut S::Inner<T> as *mut Slice<T, S>) }
    }

    #[inline]
    fn as_slice(&self) -> &Slice<T, S> {
        unsafe { &*(&self.inner as *const S::Inner<T> as *const Slice<T, S>) }
    }
}

impl<T: Clone, S: ConstShape, A: Allocator + Clone> Clone for StaticBuffer<T, S, A> {
    #[inline]
    fn clone(&self) -> Self {
        Owned::clone(self)
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        Owned::clone_from(self, source);
    }
}

impl<T: Copy, S: ConstShape<Inner<T>: Copy>, A: Allocator + Copy> Copy for StaticBuffer<T, S, A> {}

impl<T, S: ConstShape, A: Allocator> Owned for StaticBuffer<T, S, A> {
    type Alloc = A;
    type WithConst<const N: usize> = S::WithConst<T, N, A>;

    #[inline]
    fn allocator(&self) -> &A {
        &self.alloc
    }

    #[inline]
    unsafe fn cast<U>(self) -> S::Buffer<U, A> {
        assert!(Layout::new::<T>() == Layout::new::<U>(), "layout mismatch");

        let inner = unsafe { mem::transmute_copy(&ManuallyDrop::new(self.inner)) };
        let buffer = StaticBuffer::<U, S, A> { inner, alloc: self.alloc };

        buffer.into_buffer()
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    fn clone(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        let buffer = self.as_slice().expr().cloned().eval().into_inner();

        Self { inner: buffer.into_buffer::<StaticBuffer<T, S>>().inner, alloc: self.alloc.clone() }
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn clone(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        self.as_slice().expr().cloned().eval_in(self.alloc.clone()).into_inner().into_buffer()
    }

    #[inline]
    fn clone_from(&mut self, source: &Self)
    where
        T: Clone,
        A: Clone,
    {
        self.as_mut_slice()[..].clone_from_slice(&source.as_slice()[..]);
    }

    #[inline]
    fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        self.as_mut_slice()[..].clone_from_slice(&slice[..]);
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    fn from_dyn(buffer: DynBuffer<T, S, A>) -> Self {
        let (mut vec, _) = buffer.into_parts();

        let inner = unsafe { ptr::read(vec.as_ptr() as *const S::Inner<T>) };
        let alloc = unsafe { mem::transmute_copy(&Global) };

        unsafe {
            vec.set_len(0);
        }

        Self { inner, alloc }
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn from_dyn(buffer: DynBuffer<T, S, A>) -> Self {
        let (vec, _) = buffer.into_parts();
        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        let inner = unsafe { ptr::read(ptr as *const S::Inner<T>) };

        _ = unsafe { Vec::from_raw_parts_in(ptr, 0, capacity, &alloc) };

        Self { inner, alloc }
    }

    #[inline]
    fn from_static<R: ConstShape>(buffer: StaticBuffer<T, R, A>, new_shape: S) -> Self {
        assert!(new_shape.checked_len() == Some(R::default().len()), "length must not change");

        let inner = unsafe { mem::transmute_copy(&ManuallyDrop::new(buffer.inner)) };

        StaticBuffer { inner, alloc: buffer.alloc }
    }

    #[inline]
    fn into_buffer<B: Owned<Item = T, Alloc = A>>(self) -> B {
        B::from_static(self, S::default().with_dims(B::Shape::from_dims))
    }

    #[inline]
    fn into_dyn(self) -> DynBuffer<Self::Item, Self::Shape, Self::Alloc> {
        self.into_buffer()
    }

    #[inline]
    fn into_shape<R: Shape>(self, new_shape: R) -> R::Buffer<T, A> {
        Owned::from_static(self, new_shape)
    }

    #[inline]
    fn uninit_in(shape: S, alloc: A) -> S::Buffer<MaybeUninit<T>, A> {
        assert!(shape.checked_len().is_some(), "invalid length");

        let inner = unsafe { mem::transmute_copy(&<MaybeUninit<S::Inner<T>>>::uninit()) };
        let buffer = StaticBuffer::<MaybeUninit<T>, S, A> { inner, alloc };

        buffer.into_buffer()
    }

    #[inline]
    fn zip_with<U, E: Expression, F>(self, expr: E, f: F) -> S::Buffer<U, A>
    where
        F: FnMut((T, E::Item)) -> U,
    {
        let buffer = StaticBuffer::<T, S> { inner: self.inner, alloc: Global };
        let array = unsafe { IntoExpr::new(buffer.cast()) };

        <Array<U, S, A>>::with_expr_in(array.zip(expr).map(f), self.alloc).into_inner()
    }
}
