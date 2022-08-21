use std::alloc::Allocator;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{cmp, mem};

use crate::dim::Dim;
use crate::layout::{DenseLayout, Layout};

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
    vec: Vec<T, A>,
    layout: DenseLayout<D>,
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

struct VecGuard<'a, T, A: Allocator> {
    ptr: *mut T,
    len: usize,
    phantom: PhantomData<&'a mut Vec<T, A>>,
}

impl<T, D: Dim, A: Allocator> DenseBuffer<T, D, A> {
    pub(crate) unsafe fn from_parts(vec: Vec<T, A>, layout: DenseLayout<D>) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");
        assert!(D::RANK > 0, "invalid rank");

        Self { vec, layout }
    }

    pub(crate) fn into_parts(self) -> (Vec<T, A>, DenseLayout<D>) {
        let Self { vec, layout } = self;

        (vec, layout)
    }

    pub(crate) fn resize_with<F: FnMut() -> T>(&mut self, shape: D::Shape, mut f: F)
    where
        T: Clone,
        A: Clone,
    {
        let len = shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        if len == 0 {
            self.vec.clear();
        } else {
            let inner_dims = D::dims(..D::RANK - 1);
            let old_shape = self.layout.shape();

            if shape[inner_dims.clone()] == old_shape[inner_dims] {
                self.vec.resize_with(len, f);
            } else {
                let mut vec = Vec::with_capacity_in(len, self.vec.allocator().clone());

                self.layout = Layout::default(); // Remove contents in case of exception.

                unsafe {
                    copy_dim::<T, D, A, D::Lower, F>(
                        &mut VecGuard::new(&mut self.vec),
                        &mut vec,
                        old_shape,
                        shape,
                        &mut f,
                    );
                }

                self.vec = vec;
            }
        }

        self.layout = DenseLayout::new(shape);
    }

    pub(crate) unsafe fn set_layout(&mut self, layout: DenseLayout<D>) {
        self.layout = layout;
    }

    pub(crate) fn vec(&self) -> &Vec<T, A> {
        &self.vec
    }

    pub(crate) unsafe fn vec_mut(&mut self) -> &mut Vec<T, A> {
        &mut self.vec
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
        Self { vec: self.vec.clone(), layout: self.layout }
    }

    fn clone_from(&mut self, src: &Self) {
        self.layout = Layout::default();
        self.vec.clone_from(&src.vec);
        self.layout = src.layout;
    }
}

macro_rules! impl_sub_buffer {
    ($name:tt, $raw_mut:tt) => {
        impl<'a, T, L: Copy> $name<'a, T, L> {
            pub(crate) unsafe fn new_unchecked(ptr: *$raw_mut T, layout: L) -> Self {
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
        Self { ptr: self.ptr, layout: self.layout, phantom: PhantomData }
    }
}

impl<'a, T, A: Allocator> VecGuard<'a, T, A> {
    fn new(vec: &'a mut Vec<T, A>) -> Self {
        let len = vec.len();

        unsafe {
            vec.set_len(0);
        }

        Self { ptr: vec.as_mut_ptr(), len, phantom: PhantomData }
    }
}

impl<'a, T, A: Allocator> Drop for VecGuard<'a, T, A> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

unsafe fn copy_dim<T: Clone, D: Dim, A: Allocator, I: Dim, F: FnMut() -> T>(
    old_vec: &mut VecGuard<T, A>,
    new_vec: &mut Vec<T, A>,
    old_shape: D::Shape,
    new_shape: D::Shape,
    f: &mut F,
) {
    let inner_dims = D::dims(..I::RANK);

    let old_stride: usize = old_shape[inner_dims.clone()].iter().product();
    let new_stride: usize = new_shape[inner_dims].iter().product();

    let old_size = old_shape[D::dim(I::RANK)];
    let new_size = new_shape[D::dim(I::RANK)];

    let min_size = cmp::min(old_size, new_size);

    if I::RANK == 0 {
        ptr::copy(old_vec.ptr, new_vec.as_mut_ptr().add(new_vec.len()), min_size);

        old_vec.ptr = old_vec.ptr.add(min_size);
        old_vec.len -= min_size;

        new_vec.set_len(new_vec.len() + min_size);
    } else {
        for _ in 0..min_size {
            copy_dim::<T, D, A, I::Lower, F>(old_vec, new_vec, old_shape, new_shape, f);
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
