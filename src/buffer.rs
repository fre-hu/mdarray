#[cfg(feature = "nightly")]
use std::alloc::Allocator;
#[cfg(feature = "nightly")]
use std::marker::PhantomData;
#[cfg(not(feature = "nightly"))]
use std::marker::{PhantomData, PhantomPinned};
use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::{cmp, ptr};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::SpanArray;
use crate::dim::Dim;
use crate::format::{Dense, Format};
use crate::layout::{DenseLayout, Layout};
use crate::raw_span::RawSpan;

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

/// Buffer trait for array storage.
pub trait Buffer {
    /// Array element type.
    type Item;

    /// Array dimension type.
    type Dim: Dim;

    /// Array format type.
    type Format: Format;

    #[doc(hidden)]
    fn as_span(&self) -> &SpanArray<Self::Item, Self::Dim, Self::Format>;
}

/// Mutable buffer trait for array storage.
pub trait BufferMut: Buffer {
    #[doc(hidden)]
    fn as_mut_span(&mut self) -> &mut SpanArray<Self::Item, Self::Dim, Self::Format>;
}

/// Sized buffer trait for array storage.
pub trait SizedBuffer: Buffer {}

/// Mutable sized buffer trait for array storage.
pub trait SizedBufferMut: BufferMut + SizedBuffer {}

/// Dense multidimensional array buffer.
pub struct GridBuffer<T, D: Dim, A: Allocator> {
    span: RawSpan<T, D, Dense>,
    capacity: usize,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<A>,
    #[cfg(feature = "nightly")]
    alloc: ManuallyDrop<A>,
}

/// Multidimensional array span buffer.
pub struct SpanBuffer<T, D: Dim, F: Format> {
    phantom: PhantomData<(T, D, F)>,
    #[cfg(not(feature = "nightly"))]
    _pinned: PhantomPinned,
    #[cfg(feature = "nightly")]
    _opaque: Opaque,
}

/// Multidimensional array view buffer.
pub struct ViewBuffer<'a, T, D: Dim, F: Format> {
    span: RawSpan<T, D, F>,
    phantom: PhantomData<&'a T>,
}

/// Mutable multidimensional array view buffer.
pub struct ViewBufferMut<'a, T, D: Dim, F: Format> {
    span: RawSpan<T, D, F>,
    phantom: PhantomData<&'a mut T>,
}

pub struct VecGuard<'a, T, D: Dim, A: Allocator> {
    vec: ManuallyDrop<vec_t!(T, A)>,
    phantom: PhantomData<&'a GridBuffer<T, D, A>>,
}

pub struct VecGuardMut<'a, T, D: Dim, A: Allocator> {
    vec: ManuallyDrop<vec_t!(T, A)>,
    buffer: &'a mut GridBuffer<T, D, A>,
}

struct DropGuard<'a, T, A: Allocator> {
    ptr: *mut T,
    len: usize,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<(&'a mut Vec<T>, &'a A)>,
    #[cfg(feature = "nightly")]
    phantom: PhantomData<&'a mut Vec<T, A>>,
}

#[cfg(feature = "nightly")]
extern "C" {
    type Opaque;
}

impl<T, D: Dim, A: Allocator> GridBuffer<T, D, A> {
    #[cfg(feature = "nightly")]
    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) unsafe fn from_parts(vec: Vec<T>, layout: DenseLayout<D>) -> Self {
        assert!(D::RANK > 0, "invalid rank");

        let mut vec = ManuallyDrop::new(vec);

        Self {
            span: RawSpan::new_unchecked(vec.as_mut_ptr(), layout),
            capacity: vec.capacity(),
            phantom: PhantomData,
        }
    }

    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn from_parts(vec: Vec<T, A>, layout: DenseLayout<D>) -> Self {
        assert!(D::RANK > 0, "invalid rank");

        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        Self {
            span: RawSpan::new_unchecked(ptr, layout),
            capacity,
            alloc: ManuallyDrop::new(alloc),
        }
    }

    pub(crate) fn guard(&self) -> VecGuard<T, D, A> {
        VecGuard::new(self)
    }

    pub(crate) fn guard_mut(&mut self) -> VecGuardMut<T, D, A> {
        VecGuardMut::new(self)
    }

    pub(crate) fn into_parts(self) -> (vec_t!(T, A), DenseLayout<D>) {
        let mut me = mem::ManuallyDrop::new(self);

        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(me.span.as_mut_ptr(), me.span.layout().len(), me.capacity)
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                me.span.as_mut_ptr(),
                me.span.layout().len(),
                me.capacity,
                ptr::read(&*me.alloc),
            )
        };

        (vec, me.span.layout())
    }

    pub(crate) fn resize_with<F: FnMut() -> T>(&mut self, new_shape: D::Shape, mut f: F)
    where
        A: Clone,
    {
        let new_len = new_shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));

        let old_shape = self.span.layout().shape();
        let mut guard = self.guard_mut();

        if new_len == 0 {
            guard.clear();
        } else if new_shape[..D::RANK - 1] == old_shape[..D::RANK - 1] {
            guard.resize_with(new_len, f);
        } else {
            #[cfg(not(feature = "nightly"))]
            let mut vec = Vec::with_capacity(new_len);
            #[cfg(feature = "nightly")]
            let mut vec = Vec::with_capacity_in(new_len, guard.allocator().clone());

            unsafe {
                copy_dim::<T, D, A, D::Lower>(
                    &mut DropGuard::new(&mut guard),
                    &mut vec,
                    old_shape,
                    new_shape,
                    &mut f,
                );
            }

            *guard = vec;
        }

        guard.set_layout(DenseLayout::new(new_shape));
    }

    pub(crate) unsafe fn set_layout(&mut self, new_layout: DenseLayout<D>) {
        self.span.set_layout(new_layout);
    }
}

impl<T, D: Dim, A: Allocator> Buffer for GridBuffer<T, D, A> {
    type Item = T;
    type Dim = D;
    type Format = Dense;

    fn as_span(&self) -> &SpanArray<T, D, Dense> {
        self.span.as_span()
    }
}

impl<T, D: Dim, A: Allocator> BufferMut for GridBuffer<T, D, A> {
    fn as_mut_span(&mut self) -> &mut SpanArray<T, D, Dense> {
        self.span.as_mut_span()
    }
}

impl<T: Clone, D: Dim, A: Allocator + Clone> Clone for GridBuffer<T, D, A> {
    fn clone(&self) -> Self {
        unsafe { Self::from_parts(self.guard().clone(), self.span.layout()) }
    }

    fn clone_from(&mut self, source: &Self) {
        let mut guard = self.guard_mut();

        guard.clone_from(&source.guard());
        guard.set_layout(source.span.layout());
    }
}

impl<T, D: Dim, A: Allocator> Drop for GridBuffer<T, D, A> {
    fn drop(&mut self) {
        #[cfg(not(feature = "nightly"))]
        unsafe {
            Vec::from_raw_parts(self.span.as_mut_ptr(), self.span.layout().len(), self.capacity);
        }
        #[cfg(feature = "nightly")]
        unsafe {
            Vec::from_raw_parts_in(
                self.span.as_mut_ptr(),
                self.span.layout().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            );
        }
    }
}

impl<T, D: Dim, A: Allocator> SizedBuffer for GridBuffer<T, D, A> {}
impl<T, D: Dim, A: Allocator> SizedBufferMut for GridBuffer<T, D, A> {}

unsafe impl<T: Send, D: Dim, A: Allocator + Send> Send for GridBuffer<T, D, A> {}
unsafe impl<T: Sync, D: Dim, A: Allocator + Sync> Sync for GridBuffer<T, D, A> {}

impl<T, D: Dim, F: Format> Buffer for SpanBuffer<T, D, F> {
    type Item = T;
    type Dim = D;
    type Format = F;

    fn as_span(&self) -> &SpanArray<T, D, F> {
        RawSpan::from_buffer(self).as_span()
    }
}

impl<T, D: Dim, F: Format> BufferMut for SpanBuffer<T, D, F> {
    fn as_mut_span(&mut self) -> &mut SpanArray<T, D, F> {
        RawSpan::from_mut_buffer(self).as_mut_span()
    }
}

unsafe impl<T: Send, D: Dim, F: Format> Send for SpanBuffer<T, D, F> {}
unsafe impl<T: Sync, D: Dim, F: Format> Sync for SpanBuffer<T, D, F> {}

macro_rules! impl_view_buffer {
    ($name:tt, $raw_mut:tt) => {
        impl<'a, T, D: Dim, F: Format> $name<'a, T, D, F> {
            pub(crate) unsafe fn new_unchecked(ptr: *$raw_mut T, layout: Layout<D, F>) -> Self {
                Self {
                    span: RawSpan::new_unchecked(ptr as *mut T, layout),
                    phantom: PhantomData,
                }
            }
        }

        impl<'a, T, D: Dim, F: Format> Buffer for $name<'a, T, D, F> {
            type Item = T;
            type Dim = D;
            type Format = F;

            fn as_span(&self) -> &SpanArray<T, D, F> {
                self.span.as_span()
            }
        }
    };
}

impl_view_buffer!(ViewBuffer, const);
impl_view_buffer!(ViewBufferMut, mut);

impl<'a, T, D: Dim, F: Format> BufferMut for ViewBufferMut<'a, T, D, F> {
    fn as_mut_span(&mut self) -> &mut SpanArray<T, D, F> {
        self.span.as_mut_span()
    }
}

impl<'a, T, D: Dim, F: Format> Clone for ViewBuffer<'a, T, D, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, D: Dim, F: Format> Copy for ViewBuffer<'a, T, D, F> {}

impl<'a, T, D: Dim, F: Format> SizedBuffer for ViewBuffer<'a, T, D, F> {}

impl<'a, T, D: Dim, F: Format> SizedBuffer for ViewBufferMut<'a, T, D, F> {}
impl<'a, T, D: Dim, F: Format> SizedBufferMut for ViewBufferMut<'a, T, D, F> {}

unsafe impl<'a, T: Sync, D: Dim, F: Format> Send for ViewBuffer<'a, T, D, F> {}
unsafe impl<'a, T: Sync, D: Dim, F: Format> Sync for ViewBuffer<'a, T, D, F> {}

unsafe impl<'a, T: Send, D: Dim, F: Format> Send for ViewBufferMut<'a, T, D, F> {}
unsafe impl<'a, T: Sync, D: Dim, F: Format> Sync for ViewBufferMut<'a, T, D, F> {}

impl<'a, T, D: Dim, A: Allocator> VecGuard<'a, T, D, A> {
    pub fn new(buffer: &'a GridBuffer<T, D, A>) -> Self {
        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(
                buffer.span.as_ptr() as *mut T,
                buffer.span.layout().len(),
                buffer.capacity,
            )
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                buffer.span.as_ptr() as *mut T,
                buffer.span.layout().len(),
                buffer.capacity,
                ptr::read(&*buffer.alloc),
            )
        };

        Self { vec: ManuallyDrop::new(vec), phantom: PhantomData }
    }
}

impl<'a, T, D: Dim, A: Allocator> Deref for VecGuard<'a, T, D, A> {
    type Target = vec_t!(T, A);

    fn deref(&self) -> &vec_t!(T, A) {
        &self.vec
    }
}

impl<'a, T, D: Dim, A: Allocator> VecGuardMut<'a, T, D, A> {
    pub fn new(buffer: &'a mut GridBuffer<T, D, A>) -> Self {
        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(
                buffer.span.as_mut_ptr(),
                buffer.span.layout().len(),
                buffer.capacity,
            )
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                buffer.span.as_mut_ptr(),
                buffer.span.layout().len(),
                buffer.capacity,
                ptr::read(&*buffer.alloc),
            )
        };

        Self { vec: ManuallyDrop::new(vec), buffer }
    }

    pub fn set_layout(&mut self, new_layout: DenseLayout<D>) {
        unsafe {
            self.buffer.span.set_layout(new_layout);
        }
    }
}

impl<'a, T, D: Dim, A: Allocator> Deref for VecGuardMut<'a, T, D, A> {
    type Target = vec_t!(T, A);

    fn deref(&self) -> &vec_t!(T, A) {
        &self.vec
    }
}

impl<'a, T, D: Dim, A: Allocator> DerefMut for VecGuardMut<'a, T, D, A> {
    fn deref_mut(&mut self) -> &mut vec_t!(T, A) {
        &mut self.vec
    }
}

impl<'a, T, D: Dim, A: Allocator> Drop for VecGuardMut<'a, T, D, A> {
    fn drop(&mut self) {
        #[cfg(not(feature = "nightly"))]
        unsafe {
            self.buffer.span.set_ptr(self.vec.as_mut_ptr());
            self.buffer.capacity = self.vec.capacity();

            let layout = self.buffer.span.layout();

            // Cleanup in case of length mismatch (e.g. due to allocation failure)
            if self.vec.len() != layout.len() {
                if self.vec.len() > layout.len() {
                    ptr::drop_in_place(&mut self.vec.as_mut_slice()[layout.len()..]);
                } else {
                    self.buffer.span.set_layout(Layout::default());
                    ptr::drop_in_place(self.vec.as_mut_slice());
                }
            }
        }
        #[cfg(feature = "nightly")]
        unsafe {
            let (ptr, len, capacity, alloc) = ptr::read(&*self.vec).into_raw_parts_with_alloc();

            self.buffer.span.set_ptr(ptr);
            self.buffer.capacity = capacity;
            self.buffer.alloc = ManuallyDrop::new(alloc);

            let layout = self.buffer.span.layout();

            // Cleanup in case of length mismatch (e.g. due to allocation failure)
            if len != layout.len() {
                if len > layout.len() {
                    ptr::drop_in_place(&mut self.vec.as_mut_slice()[layout.len()..]);
                } else {
                    self.buffer.span.set_layout(Layout::default());
                    ptr::drop_in_place(self.vec.as_mut_slice());
                }
            }
        }
    }
}

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
    let old_stride: usize = old_shape[..I::RANK].iter().product();
    let new_stride: usize = new_shape[..I::RANK].iter().product();

    let old_size = old_shape[I::RANK];
    let new_size = new_shape[I::RANK];

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
