#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::{cmp, ptr};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::index::{Axis, Outer};
use crate::layout::Dense;
use crate::mapping::{DenseMapping, Mapping};
use crate::raw_span::RawSpan;
use crate::shape::Shape;
use crate::span::Span;

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

pub(crate) struct RawGrid<T, S: Shape, A: Allocator> {
    span: RawSpan<T, S, Dense>,
    capacity: usize,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<A>,
    #[cfg(feature = "nightly")]
    alloc: ManuallyDrop<A>,
}

struct DropGuard<'a, T, A: Allocator> {
    ptr: *mut T,
    len: usize,
    #[cfg(not(feature = "nightly"))]
    phantom: PhantomData<(&'a mut Vec<T>, &'a A)>,
    #[cfg(feature = "nightly")]
    phantom: PhantomData<&'a mut Vec<T, A>>,
}

impl<T, S: Shape, A: Allocator> RawGrid<T, S, A> {
    #[cfg(feature = "nightly")]
    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) fn as_mut_span(&mut self) -> &mut Span<T, S> {
        self.span.as_mut_span()
    }

    pub(crate) fn as_span(&self) -> &Span<T, S> {
        self.span.as_span()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) unsafe fn from_parts(vec: Vec<T>, mapping: DenseMapping<S>) -> Self {
        debug_assert!(Some(vec.len()) == mapping.shape().checked_len(), "length mismatch");

        let mut vec = ManuallyDrop::new(vec);

        Self {
            span: RawSpan::new_unchecked(vec.as_mut_ptr(), mapping),
            capacity: vec.capacity(),
            phantom: PhantomData,
        }
    }

    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn from_parts(vec: Vec<T, A>, mapping: DenseMapping<S>) -> Self {
        debug_assert!(Some(vec.len()) == mapping.shape().checked_len(), "length mismatch");

        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        Self {
            span: RawSpan::new_unchecked(ptr, mapping),
            capacity,
            alloc: ManuallyDrop::new(alloc),
        }
    }

    pub(crate) fn into_parts(self) -> (vec_t!(T, A), DenseMapping<S>) {
        let mut me = ManuallyDrop::new(self);

        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(me.span.as_mut_ptr(), me.span.mapping().len(), me.capacity)
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                me.span.as_mut_ptr(),
                me.span.mapping().len(),
                me.capacity,
                ptr::read(&*me.alloc),
            )
        };

        (vec, me.span.mapping())
    }

    pub(crate) fn resize_with<F: FnMut() -> T>(&mut self, new_shape: S, mut f: F)
    where
        A: Clone,
    {
        if S::RANK > 0 {
            let new_len = new_shape.checked_len().expect("invalid length");
            let old_shape = self.span.mapping().shape();

            unsafe {
                self.with_mut_vec(|vec| {
                    if new_len == 0 {
                        vec.clear();
                    } else if new_shape.dims()[..S::RANK - 1] == old_shape.dims()[..S::RANK - 1] {
                        vec.resize_with(new_len, f);
                    } else {
                        #[cfg(not(feature = "nightly"))]
                        let mut new_vec = Vec::with_capacity(new_len);
                        #[cfg(feature = "nightly")]
                        let mut new_vec = Vec::with_capacity_in(new_len, vec.allocator().clone());

                        copy_dim::<T, S, A>(
                            &mut DropGuard::new(vec),
                            &mut new_vec,
                            old_shape,
                            new_shape,
                            &mut f,
                        );

                        *vec = new_vec;
                    }
                });

                self.set_mapping(DenseMapping::new(new_shape));
            }
        }
    }

    pub(crate) unsafe fn set_mapping(&mut self, new_mapping: DenseMapping<S>) {
        debug_assert!(new_mapping.len() <= self.capacity, "length exceeds capacity");

        self.span.set_mapping(new_mapping);
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) unsafe fn with_mut_vec<U, F: FnOnce(&mut Vec<T>) -> U>(&mut self, f: F) -> U {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            grid: &'a mut RawGrid<T, S, A>,
            vec: ManuallyDrop<Vec<T>>,
        }

        impl<'a, T, S: Shape, A: Allocator> Drop for DropGuard<'a, T, S, A> {
            fn drop(&mut self) {
                unsafe {
                    self.grid.span.set_ptr(self.vec.as_mut_ptr());
                    self.grid.capacity = self.vec.capacity();

                    let mapping = self.grid.span.mapping();

                    // Cleanup in case of length mismatch (e.g. due to allocation failure)
                    if self.vec.len() != mapping.len() {
                        if self.vec.len() > mapping.len() {
                            ptr::drop_in_place(&mut self.vec.as_mut_slice()[mapping.len()..]);
                        } else {
                            self.grid.span.set_mapping(DenseMapping::default());
                            ptr::drop_in_place(self.vec.as_mut_slice());
                        }
                    }
                }
            }
        }

        let vec =
            Vec::from_raw_parts(self.span.as_mut_ptr(), self.span.mapping().len(), self.capacity);

        let mut guard = DropGuard { grid: self, vec: ManuallyDrop::new(vec) };

        let result = f(&mut guard.vec);

        guard.grid.span.set_ptr(guard.vec.as_mut_ptr());
        guard.grid.capacity = guard.vec.capacity();

        mem::forget(guard);

        result
    }

    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn with_mut_vec<U, F: FnOnce(&mut Vec<T, A>) -> U>(&mut self, f: F) -> U {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            grid: &'a mut RawGrid<T, S, A>,
            vec: ManuallyDrop<Vec<T, A>>,
        }

        impl<'a, T, S: Shape, A: Allocator> Drop for DropGuard<'a, T, S, A> {
            fn drop(&mut self) {
                unsafe {
                    self.grid.span.set_ptr(self.vec.as_mut_ptr());
                    self.grid.capacity = self.vec.capacity();
                    self.grid.alloc = ManuallyDrop::new(ptr::read(self.vec.allocator()));

                    let mapping = self.grid.span.mapping();

                    // Cleanup in case of length mismatch (e.g. due to allocation failure)
                    if self.vec.len() != mapping.len() {
                        if self.vec.len() > mapping.len() {
                            ptr::drop_in_place(&mut self.vec.as_mut_slice()[mapping.len()..]);
                        } else {
                            self.grid.span.set_mapping(DenseMapping::default());
                            ptr::drop_in_place(self.vec.as_mut_slice());
                        }
                    }
                }
            }
        }

        let vec = unsafe {
            Vec::from_raw_parts_in(
                self.span.as_mut_ptr(),
                self.span.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };

        let mut guard = DropGuard { grid: self, vec: ManuallyDrop::new(vec) };

        let result = f(&mut guard.vec);

        guard.grid.span.set_ptr(guard.vec.as_mut_ptr());
        guard.grid.capacity = guard.vec.capacity();
        guard.grid.alloc = ManuallyDrop::new(ptr::read(guard.vec.allocator()));

        mem::forget(guard);

        result
    }

    pub(crate) fn with_vec<U, F: FnOnce(&vec_t!(T, A)) -> U>(&self, f: F) -> U {
        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(
                self.span.as_ptr() as *mut T,
                self.span.mapping().len(),
                self.capacity,
            )
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                self.span.as_ptr() as *mut T,
                self.span.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };

        f(&ManuallyDrop::new(vec))
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for RawGrid<T, S, A> {
    fn clone(&self) -> Self {
        unsafe { Self::from_parts(self.with_vec(|vec| vec.clone()), self.span.mapping()) }
    }

    fn clone_from(&mut self, source: &Self) {
        unsafe {
            self.with_mut_vec(|dst| source.with_vec(|src| dst.clone_from(src)));
            self.set_mapping(source.span.mapping());
        }
    }
}

impl<T, S: Shape, A: Allocator> Drop for RawGrid<T, S, A> {
    #[cfg(not(feature = "nightly"))]
    fn drop(&mut self) {
        _ = unsafe {
            Vec::from_raw_parts(self.span.as_mut_ptr(), self.span.mapping().len(), self.capacity)
        };
    }

    #[cfg(feature = "nightly")]
    fn drop(&mut self) {
        _ = unsafe {
            Vec::from_raw_parts_in(
                self.span.as_mut_ptr(),
                self.span.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };
    }
}

unsafe impl<T: Send, S: Shape, A: Allocator + Send> Send for RawGrid<T, S, A> {}
unsafe impl<T: Sync, S: Shape, A: Allocator + Sync> Sync for RawGrid<T, S, A> {}

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

unsafe fn copy_dim<T, S: Shape, A: Allocator>(
    old_vec: &mut DropGuard<T, A>,
    new_vec: &mut vec_t!(T, A),
    old_shape: S,
    new_shape: S,
    f: &mut impl FnMut() -> T,
) {
    let old_stride: usize = old_shape.dims()[..S::RANK - 1].iter().product();
    let new_stride: usize = new_shape.dims()[..S::RANK - 1].iter().product();

    let old_size = old_shape.dim(S::RANK - 1);
    let new_size = new_shape.dim(S::RANK - 1);

    let min_size = cmp::min(old_size, new_size);

    if S::RANK > 1 {
        for _ in 0..min_size {
            copy_dim::<T, <Outer as Axis>::Other<S>, A>(
                old_vec,
                new_vec,
                old_shape.remove_dim(S::RANK - 1),
                new_shape.remove_dim(S::RANK - 1),
                f,
            );
        }
    } else {
        debug_assert!(old_vec.len >= min_size, "slice exceeds remainder");
        debug_assert!(new_vec.len() + min_size <= new_vec.capacity(), "slice exceeds capacity");

        ptr::copy_nonoverlapping(old_vec.ptr, new_vec.as_mut_ptr().add(new_vec.len()), min_size);

        old_vec.ptr = old_vec.ptr.add(min_size);
        old_vec.len -= min_size;

        new_vec.set_len(new_vec.len() + min_size);
    }

    if old_size > min_size {
        let count = (old_size - min_size) * old_stride;
        let slice = ptr::slice_from_raw_parts_mut(old_vec.ptr, count);

        debug_assert!(old_vec.len >= count, "slice exceeds remainder");

        old_vec.ptr = old_vec.ptr.add(count);
        old_vec.len -= count;

        ptr::drop_in_place(slice);
    }

    let additional = (new_size - min_size) * new_stride;

    debug_assert!(new_vec.len() + additional <= new_vec.capacity(), "slice exceeds capacity");

    for _ in 0..additional {
        new_vec.as_mut_ptr().add(new_vec.len()).write(f());
        new_vec.set_len(new_vec.len() + 1);
    }
}
