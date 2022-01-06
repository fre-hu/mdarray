use std::alloc::Allocator;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{cmp, mem};

use crate::dimension::Dim;
use crate::layout::{DenseLayout, Layout};
use crate::order::Order;
use crate::raw_vec::RawVec;

pub trait Buffer {
    type Item;
    type Layout: Layout;

    fn as_ptr(&self) -> *const Self::Item;
    fn layout(&self) -> &Self::Layout;
}

pub trait BufferMut: Buffer {
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
}

pub struct DenseBuffer<T, D: Dim, O: Order, A: Allocator> {
    vec: RawVec<T, A>,
    layout: DenseLayout<D, O>,
}

pub struct SubBuffer<'a, T, L: Layout> {
    ptr: NonNull<T>,
    layout: L,
    _marker: PhantomData<&'a T>,
}

pub struct SubBufferMut<'a, T, L: Layout> {
    ptr: NonNull<T>,
    layout: L,
    _marker: PhantomData<&'a mut T>,
}

impl<T, D: Dim, O: Order, A: Allocator> DenseBuffer<T, D, O, A> {
    pub fn allocator(&self) -> &A {
        self.vec.allocator()
    }

    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    pub fn clear(&mut self) {
        let len = self.layout.len();

        self.layout = DenseLayout::default();

        for i in 0..len {
            unsafe {
                ptr::drop_in_place(self.as_mut_ptr().add(i));
            }
        }
    }

    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: D::Shape,
        capacity: usize,
        alloc: A,
    ) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self {
            vec: RawVec::from_raw_parts_in(ptr, capacity, alloc),
            layout: DenseLayout::new(shape),
        }
    }

    pub fn into_raw_parts_with_alloc(self) -> (*mut T, D::Shape, usize, A) {
        let mut me = mem::ManuallyDrop::new(self);

        (me.as_mut_ptr(), me.layout.shape(), me.capacity(), unsafe { ptr::read(me.allocator()) })
    }

    pub fn new_in(alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { vec: RawVec::new_in(alloc), layout: DenseLayout::default() }
    }

    pub fn resize(&mut self, shape: D::Shape, value: T)
    where
        T: Clone,
        A: Clone,
    {
        let new_len = shape.as_ref().iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        if new_len == 0 {
            self.clear();
        } else {
            let old_len = self.layout.len();
            let old_shape = self.layout.shape();

            let copy = O::select(
                shape.as_ref()[..D::RANK - 1] != old_shape.as_ref()[..D::RANK - 1],
                shape.as_ref()[1..] != old_shape.as_ref()[1..],
            );

            if copy {
                let mut vec = RawVec::with_capacity_in(new_len, self.allocator().clone());

                self.layout = DenseLayout::default(); // Leak elements in case of exception.

                unsafe {
                    Self::copy_dim(
                        self.as_mut_ptr(),
                        vec.as_mut_ptr(),
                        old_shape,
                        shape,
                        &value,
                        O::select(D::RANK - 1, 0),
                    );
                }

                mem::swap(&mut self.vec, &mut vec);

                self.layout = DenseLayout::new(shape);
            } else {
                if new_len > self.capacity() {
                    self.vec.grow(new_len);
                }

                let ptr = self.as_mut_ptr();

                for i in old_len..new_len {
                    unsafe {
                        ptr::write(ptr.add(i), value.clone());
                    }
                }

                self.layout = DenseLayout::new(shape); // Resize after creating new elements.

                for i in new_len..old_len {
                    unsafe {
                        ptr::drop_in_place(ptr.add(i));
                    }
                }
            }
        }
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        if capacity < self.capacity() {
            self.vec.shrink(capacity);
        }
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        Self { vec: RawVec::with_capacity_in(capacity, alloc), layout: DenseLayout::default() }
    }

    unsafe fn copy_dim(
        old_ptr: *mut T,
        new_ptr: *mut T,
        old_shape: D::Shape,
        new_shape: D::Shape,
        value: &T,
        dim: usize,
    ) where
        T: Clone,
    {
        let old_stride: usize = O::select(
            old_shape.as_ref()[..dim].iter().product(),
            old_shape.as_ref()[dim + 1..].iter().product(),
        );

        let new_stride: usize = O::select(
            new_shape.as_ref()[..dim].iter().product(),
            new_shape.as_ref()[dim + 1..].iter().product(),
        );

        let min_size = cmp::min(old_shape.as_ref()[dim], new_shape.as_ref()[dim]);

        if dim == O::select(0, D::RANK - 1) {
            ptr::copy(old_ptr, new_ptr, min_size);
        } else {
            for i in 0..min_size {
                Self::copy_dim(
                    old_ptr.add(i * old_stride),
                    new_ptr.add(i * new_stride),
                    old_shape,
                    new_shape,
                    value,
                    O::select(dim, dim + 2) - 1,
                );
            }
        }

        for i in min_size * old_stride..old_shape.as_ref()[dim] * old_stride {
            ptr::drop_in_place(old_ptr.add(i));
        }

        for i in min_size * new_stride..new_shape.as_ref()[dim] * new_stride {
            ptr::write(new_ptr.add(i), value.clone());
        }
    }
}

impl<T, D: Dim, O: Order, A: Allocator> Buffer for DenseBuffer<T, D, O, A> {
    type Item = T;
    type Layout = DenseLayout<D, O>;

    fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &self.layout
    }
}

impl<T, D: Dim, O: Order, A: Allocator> BufferMut for DenseBuffer<T, D, O, A> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }
}

impl<T: Clone, D: Dim, O: Order, A: Allocator + Clone> Clone for DenseBuffer<T, D, O, A> {
    fn clone(&self) -> Self {
        let len = self.layout.len();

        let mut vec = RawVec::<T, A>::with_capacity_in(len, self.allocator().clone());

        for i in 0..len {
            unsafe {
                ptr::write(vec.as_mut_ptr().add(i), (*self.as_ptr().add(i)).clone());
            }
        }

        Self { vec, layout: self.layout }
    }
}

impl<T, D: Dim, O: Order, A: Allocator> Drop for DenseBuffer<T, D, O, A> {
    fn drop(&mut self) {
        self.clear();
    }
}

macro_rules! impl_sub_buffer {
    ($type:tt, $raw_mut:tt) => {
        impl<'a, T, L: Layout> $type<'a, T, L> {
            pub unsafe fn new(ptr: *$raw_mut T, layout: L) -> Self {
                assert!(mem::size_of::<T>() != 0, "ZST not allowed");

                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    layout,
                    _marker: PhantomData,
                }
            }
        }

        impl<'a, T, L: Layout> Buffer for $type<'a, T, L> {
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

impl<'a, T, L: Layout> BufferMut for SubBufferMut<'a, T, L> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<'a, T, L: Layout> Clone for SubBuffer<'a, T, L> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, layout: self.layout, _marker: PhantomData }
    }
}
