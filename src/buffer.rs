use crate::dimension::Dimension;
use crate::iterator::Drain;
use crate::layout::{DenseLayout, Layout, StridedLayout};
use crate::order::Order;
use crate::raw_vec::RawVec;
use std::alloc::Allocator;
use std::iter::TrustedLen;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{array, mem, vec};

pub trait Buffer<T, const N: usize, O: Order> {
    type Layout: Layout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T;
    fn as_ptr(&self) -> *const T;
    fn layout(&self) -> &Self::Layout;
}

pub trait FromIterIn<I: Iterator, A: Allocator> {
    fn from_iter_in(iter: I, alloc: A) -> Self;
}

pub trait OwnedBuffer<T> {
    type IntoIter: Iterator<Item = T>;
}

pub struct DenseBuffer<T, const N: usize, O: Order, A: Allocator> {
    vec: RawVec<T, A>,
    layout: DenseLayout<N, O>,
}

pub struct StaticBuffer<T, D: Dimension<N>, const N: usize, O: Order>
where
    [T; D::LEN]: ,
{
    array: [T; D::LEN],
    _marker: PhantomData<(D, O)>,
}

pub struct SubBuffer<'a, T, const N: usize, const M: usize, O: Order> {
    ptr: NonNull<T>,
    layout: StridedLayout<N, M, O>,
    _marker: PhantomData<&'a T>,
}

pub struct SubBufferMut<'a, T, const N: usize, const M: usize, O: Order> {
    ptr: NonNull<T>,
    layout: StridedLayout<N, M, O>,
    _marker: PhantomData<&'a mut T>,
}

impl<T, const N: usize, O: Order, A: Allocator> DenseBuffer<T, N, O, A> {
    pub fn allocator(&self) -> &A {
        self.vec.allocator()
    }

    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    pub fn clear(&mut self) {
        let len = self.layout.len();

        self.layout.resize([0; N], [0; 0]);

        for i in 0..len {
            unsafe {
                ptr::drop_in_place(self.as_mut_ptr().add(i));
            }
        }
    }

    pub fn drain(&mut self) -> Drain<T> {
        let len = self.layout.len();

        self.layout.resize([0; N], [0; 0]);

        Drain::new(self.as_ptr(), len)
    }

    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: [usize; N],
        capacity: usize,
        alloc: A,
    ) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            vec: RawVec::from_raw_parts_in(ptr, capacity, alloc),
            layout: DenseLayout::new(shape, []),
        }
    }

    pub fn into_raw_parts_with_alloc(self) -> (*mut T, [usize; N], usize, A) {
        let mut me = mem::ManuallyDrop::new(self);

        (me.as_mut_ptr(), me.layout.shape(), me.capacity(), unsafe {
            ptr::read(me.allocator())
        })
    }

    pub fn new_in(alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            vec: RawVec::new_in(alloc),
            layout: DenseLayout::new([0; N], []),
        }
    }

    pub fn shrink_to(&mut self, capacity: usize) {
        if capacity < self.capacity() {
            self.vec.shrink(capacity);
        }
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            vec: RawVec::with_capacity_in(capacity, alloc),
            layout: DenseLayout::new([0; N], []),
        }
    }
}

impl<T: Clone, const N: usize, O: Order, A: Allocator> DenseBuffer<T, N, O, A> {
    pub fn resize(&mut self, shape: [usize; N], value: T) {
        let old_len = self.layout.len();
        let new_len = shape.iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        if new_len > self.capacity() {
            self.vec.grow(new_len);
        }

        let ptr = self.as_mut_ptr();

        if new_len == 0 {
            self.clear();
        } else if old_len == 0 {
            for i in 0..new_len {
                unsafe {
                    ptr::write(ptr.add(i), value.clone());
                }
            }

            self.layout.resize(shape, []);
        } else {
            let mut min_shape = self.layout.shape();

            let mut count = 1;
            let mut stride = old_len;

            self.layout.resize([0; N], []); // Leak elements in case of exception

            // Shrink dimensions that are too large
            for i in 0..N {
                let dim = O::select(N - 1 - i, i);

                stride /= min_shape[dim];

                if min_shape[dim] > shape[dim] {
                    let old_stride = stride * min_shape[dim];
                    let new_stride = stride * shape[dim];

                    unsafe {
                        for j in new_stride..old_stride {
                            ptr::drop_in_place(ptr.add(j));
                        }

                        for j in 1..count {
                            ptr::copy(ptr.add(j * old_stride), ptr.add(j * new_stride), new_stride);

                            for k in new_stride..old_stride {
                                ptr::drop_in_place(ptr.add(j * old_stride + k));
                            }
                        }
                    }

                    min_shape[dim] = shape[dim];
                }

                count *= min_shape[dim];
            }

            // Expand dimensions that are too small
            for i in 0..N {
                let dim = O::select(i, N - 1 - i);

                count /= min_shape[dim];

                if shape[dim] > min_shape[dim] {
                    let old_stride = stride * min_shape[dim];
                    let new_stride = stride * shape[dim];

                    unsafe {
                        for j in (1..count).rev() {
                            ptr::copy(ptr.add(j * old_stride), ptr.add(j * new_stride), old_stride);

                            for k in old_stride..new_stride {
                                ptr::write(ptr.add(j * new_stride + k), value.clone());
                            }
                        }

                        for j in old_stride..new_stride {
                            ptr::write(ptr.add(j), value.clone());
                        }
                    }
                }

                stride *= shape[dim];
            }

            self.layout.resize(shape, []);
        }
    }
}

impl<T, D: Dimension<N>, const N: usize, O: Order> StaticBuffer<T, D, N, O>
where
    [T; D::LEN]: ,
{
    const LAYOUT: DenseLayout<N, O> = DenseLayout::new(D::SHAPE, []);
}

impl<T: Copy, D: Dimension<N>, const N: usize, O: Order> StaticBuffer<T, D, N, O>
where
    [T; D::LEN]: ,
{
    pub fn new(value: T) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            array: [value; D::LEN], // TODO: Change to array and remove T: Copy
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubBuffer<'a, T, N, M, O> {
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            ptr,
            layout,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> SubBufferMut<'a, T, N, M, O> {
    pub fn new(ptr: NonNull<T>, layout: StridedLayout<N, M, O>) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        Self {
            ptr,
            layout,
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> Buffer<T, N, O> for DenseBuffer<T, N, O, A> {
    type Layout = DenseLayout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.vec.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &self.layout
    }
}

impl<T, D: Dimension<N>, const N: usize, O: Order> Buffer<T, N, O> for StaticBuffer<T, D, N, O>
where
    [(); D::LEN]: ,
{
    type Layout = DenseLayout<N, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.array.as_mut_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.array.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &Self::LAYOUT
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> Buffer<T, N, O>
    for SubBuffer<'a, T, N, M, O>
{
    type Layout = StridedLayout<N, M, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &self.layout
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> Buffer<T, N, O>
    for SubBufferMut<'a, T, N, M, O>
{
    type Layout = StridedLayout<N, M, O>;

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    fn layout(&self) -> &Self::Layout {
        &self.layout
    }
}

impl<I: Iterator<Item = T>, T, O: Order, A: Allocator> FromIterIn<I, A>
    for DenseBuffer<T, 1, O, A>
{
    default fn from_iter_in(mut iter: I, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        let mut len = 0;
        let mut vec = RawVec::<T, A>::new_in(alloc);

        while let Some(x) = iter.next() {
            if len == vec.capacity() {
                let (lower, _) = iter.size_hint();

                vec.grow(vec.capacity().saturating_add(lower).saturating_add(1));
            }

            unsafe {
                ptr::write(vec.as_mut_ptr().add(len), x);
            }

            len += 1;
        }

        Self {
            vec,
            layout: DenseLayout::new([len], []),
        }
    }
}

impl<I: TrustedLen<Item = T>, T, O: Order, A: Allocator> FromIterIn<I, A>
    for DenseBuffer<T, 1, O, A>
{
    fn from_iter_in(iter: I, alloc: A) -> Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        let (lower, _) = iter.size_hint();
        let mut vec = RawVec::<T, A>::with_capacity_in(lower, alloc);

        for (i, x) in iter.enumerate() {
            unsafe {
                ptr::write(vec.as_mut_ptr().add(i), x);
            }
        }

        Self {
            vec,
            layout: DenseLayout::new([lower], []),
        }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> OwnedBuffer<T> for DenseBuffer<T, N, O, A> {
    type IntoIter = vec::IntoIter<T, A>;
}

impl<T, D: Dimension<N>, const N: usize, O: Order> OwnedBuffer<T> for StaticBuffer<T, D, N, O>
where
    [(); D::LEN]: ,
{
    type IntoIter = array::IntoIter<T, { D::LEN }>;
}

impl<T: Clone, const N: usize, O: Order, A: Allocator + Clone> Clone for DenseBuffer<T, N, O, A> {
    fn clone(&self) -> Self {
        let len = self.layout.len();
        let mut vec = RawVec::<T, A>::with_capacity_in(len, self.allocator().clone());

        for i in 0..len {
            unsafe {
                ptr::write(vec.as_mut_ptr().add(i), (*self.as_ptr().add(i)).clone());
            }
        }

        Self {
            vec,
            layout: DenseLayout::new(self.layout.shape(), []),
        }
    }
}

impl<T: Clone, D: Dimension<N>, const N: usize, O: Order> Clone for StaticBuffer<T, D, N, O>
where
    [(); D::LEN]: ,
{
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, const N: usize, O: Order, A: Allocator> Drop for DenseBuffer<T, N, O, A> {
    fn drop(&mut self) {
        self.clear();
    }
}
