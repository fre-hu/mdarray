#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{cmp, mem};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::dim::Dim;
use crate::layout::{DenseLayout, Layout};

#[cfg(not(feature = "nightly"))]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type>
    };
}

#[cfg(feature = "nightly")]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type, $alloc>
    };
}

pub trait Buffer {
    type Item;
    type Layout: Copy;

    fn as_ptr(&self) -> *const Self::Item;
    fn layout(&self) -> &Self::Layout;
}

pub trait BufferMut: Buffer {
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
}

pub struct DenseBuffer<T, D: Dim, A: Allocator> {
    vec: vec_t!(T, A),
    layout: DenseLayout<D>,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<A>,
}

pub struct SubBuffer<'a, T, L: Copy> {
    ptr: NonNull<T>,
    layout: L,
    phantom: PhantomData<&'a T>,
}

pub struct SubBufferMut<'a, T, L: Copy> {
    ptr: NonNull<T>,
    layout: L,
    phantom: PhantomData<&'a mut T>,
}

struct DropGuard<'a, T, A: Allocator> {
    ptr: *mut T,
    len: usize,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<(&'a mut Vec<T>, &'a A)>,
    #[cfg(feature = "nightly")]
    phantom: PhantomData<&'a mut Vec<T, A>>,
}

impl<T, D: Dim, A: Allocator> DenseBuffer<T, D, A> {
    pub unsafe fn as_mut_vec(&mut self) -> &mut vec_t!(T, A) {
        &mut self.vec
    }

    pub fn as_vec(&self) -> &vec_t!(T, A) {
        &self.vec
    }

    pub unsafe fn from_parts(vec: vec_t!(T, A), layout: DenseLayout<D>) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");
        assert!(D::RANK > 0, "invalid rank");

        Self {
            vec,
            layout,
            #[cfg(not(feature = "nightly"))]
            phantom: PhantomData,
        }
    }

    pub fn into_parts(self) -> (vec_t!(T, A), DenseLayout<D>) {
        #[cfg(not(feature = "nightly"))]
        let Self { vec, layout, .. } = self;
        #[cfg(feature = "nightly")]
        let Self { vec, layout } = self;

        (vec, layout)
    }

    pub fn resize_with<F: FnMut() -> T>(&mut self, new_shape: D::Shape, mut f: F)
    where
        A: Clone,
    {
        let new_len = new_shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        if new_len == 0 {
            self.layout = DenseLayout::new(new_shape);
            self.vec.clear();
        } else {
            let inner_dims = D::dims(..D::RANK - 1);
            let old_shape = self.layout.shape();

            self.layout = Layout::default(); // Remove contents in case of exception.

            if new_shape[inner_dims.clone()] == old_shape[inner_dims] {
                self.vec.resize_with(new_len, f);
            } else {
                #[cfg(not(feature = "nightly"))]
                let mut vec = Vec::with_capacity(new_len);
                #[cfg(feature = "nightly")]
                let mut vec = Vec::with_capacity_in(new_len, self.vec.allocator().clone());

                unsafe {
                    copy_dim::<T, D, A, D::Lower>(
                        &mut DropGuard::new(&mut self.vec),
                        &mut vec,
                        old_shape,
                        new_shape,
                        &mut f,
                    );
                }

                self.vec = vec;
            }

            self.layout = DenseLayout::new(new_shape);
        }
    }

    pub unsafe fn set_layout(&mut self, new_layout: DenseLayout<D>) {
        self.layout = new_layout;
    }
}

impl<T, D: Dim, A: Allocator> Buffer for DenseBuffer<T, D, A> {
    type Item = T;
    type Layout = DenseLayout<D>;

    fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &self.layout
    }
}

impl<T, D: Dim, A: Allocator> BufferMut for DenseBuffer<T, D, A> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }
}

impl<T: Clone, D: Dim, A: Allocator + Clone> Clone for DenseBuffer<T, D, A> {
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
            layout: self.layout,
            #[cfg(not(feature = "nightly"))]
            phantom: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.layout = Layout::default(); // Remove contents in case of exception.
        self.vec.clone_from(&source.vec);
        self.layout = source.layout;
    }
}

macro_rules! impl_sub_buffer {
    ($name:tt, $raw_mut:tt) => {
        impl<'a, T, L: Copy> $name<'a, T, L> {
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, layout: L) -> Self {
                assert!(mem::size_of::<T>() != 0, "ZST not allowed");

                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    layout,
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, T, L: Copy> Buffer for $name<'a, T, L> {
            type Item = T;
            type Layout = L;

            fn as_ptr(&self) -> *const T {
                self.ptr.as_ptr()
            }

            fn layout(&self) -> &L {
                &self.layout
            }
        }
    };
}

impl_sub_buffer!(SubBuffer, const);
impl_sub_buffer!(SubBufferMut, mut);

impl<'a, T, L: Copy> BufferMut for SubBufferMut<'a, T, L> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'a, T, L: Copy> Clone for SubBuffer<'a, T, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, L: Copy> Copy for SubBuffer<'a, T, L> {}

unsafe impl<'a, T: Sync, L: Copy> Send for SubBuffer<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Copy> Sync for SubBuffer<'a, T, L> {}

unsafe impl<'a, T: Send, L: Copy> Send for SubBufferMut<'a, T, L> {}
unsafe impl<'a, T: Sync, L: Copy> Sync for SubBufferMut<'a, T, L> {}

impl<'a, T, A: Allocator> DropGuard<'a, T, A> {
    fn new(vec: &'a mut vec_t!(T, A)) -> Self {
        let len = vec.len();

        unsafe {
            vec.set_len(0);
        }

        Self { ptr: vec.as_mut_ptr(), len, phantom: PhantomData }
    }
}

impl<'a, T, A: Allocator> Drop for DropGuard<'a, T, A> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

unsafe fn copy_dim<T, D: Dim, A: Allocator, I: Dim>(
    old_vec: &mut DropGuard<T, A>,
    new_vec: &mut vec_t!(T, A),
    old_shape: D::Shape,
    new_shape: D::Shape,
    f: &mut impl FnMut() -> T,
) {
    let inner_dims = D::dims(..I::RANK);

    let old_stride: usize = old_shape[inner_dims.clone()].iter().product();
    let new_stride: usize = new_shape[inner_dims].iter().product();

    let old_size = old_shape[D::dim(I::RANK)];
    let new_size = new_shape[D::dim(I::RANK)];

    let min_size = cmp::min(old_size, new_size);

    if I::RANK == 0 {
        ptr::copy_nonoverlapping(old_vec.ptr, new_vec.as_mut_ptr().add(new_vec.len()), min_size);

        old_vec.ptr = old_vec.ptr.add(min_size);
        old_vec.len -= min_size;

        new_vec.set_len(new_vec.len() + min_size);
    } else {
        for _ in 0..min_size {
            copy_dim::<T, D, A, I::Lower>(old_vec, new_vec, old_shape, new_shape, f);
        }
    }

    if old_size > min_size {
        let count = (old_size - min_size) * old_stride;
        let slice = ptr::slice_from_raw_parts_mut(old_vec.ptr, count);

        old_vec.ptr = old_vec.ptr.add(count);
        old_vec.len -= count;

        ptr::drop_in_place(slice);
    }

    for _ in 0..(new_size - min_size) * new_stride {
        new_vec.as_mut_ptr().add(new_vec.len()).write(f());
        new_vec.set_len(new_vec.len() + 1);
    }
}
