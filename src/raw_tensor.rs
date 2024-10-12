#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::ptr;

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::layout::Dense;
use crate::mapping::{DenseMapping, Mapping};
use crate::raw_slice::RawSlice;
use crate::shape::Shape;
use crate::slice::Slice;

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

pub(crate) struct RawTensor<T, S: Shape, A: Allocator> {
    slice: RawSlice<T, S, Dense>,
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

impl<T, S: Shape, A: Allocator> RawTensor<T, S, A> {
    #[cfg(feature = "nightly")]
    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut Slice<T, S> {
        self.slice.as_mut_slice()
    }

    pub(crate) fn as_slice(&self) -> &Slice<T, S> {
        self.slice.as_slice()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) unsafe fn from_parts(vec: Vec<T>, mapping: DenseMapping<S>) -> Self {
        debug_assert!(Some(vec.len()) == mapping.shape().checked_len(), "length mismatch");

        let mut vec = ManuallyDrop::new(vec);

        Self {
            slice: RawSlice::new_unchecked(vec.as_mut_ptr(), mapping),
            capacity: vec.capacity(),
            phantom: PhantomData,
        }
    }

    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn from_parts(vec: Vec<T, A>, mapping: DenseMapping<S>) -> Self {
        debug_assert!(Some(vec.len()) == mapping.shape().checked_len(), "length mismatch");

        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        Self {
            slice: RawSlice::new_unchecked(ptr, mapping),
            capacity,
            alloc: ManuallyDrop::new(alloc),
        }
    }

    pub(crate) fn into_parts(self) -> (vec_t!(T, A), DenseMapping<S>) {
        let mut me = ManuallyDrop::new(self);

        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(me.slice.as_mut_ptr(), me.slice.mapping().len(), me.capacity)
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                me.slice.as_mut_ptr(),
                me.slice.mapping().len(),
                me.capacity,
                ptr::read(&*me.alloc),
            )
        };

        unsafe { (vec, ptr::read(me.slice.mapping())) }
    }

    pub(crate) fn resize_with<F: FnMut() -> T>(&mut self, new_dims: &[usize], mut f: F)
    where
        A: Clone,
    {
        assert!(new_dims.len() == self.slice.mapping().rank(), "invalid rank");

        if !new_dims.is_empty() {
            let new_len = new_dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
            let new_len = new_len.expect("invalid length");

            unsafe {
                self.with_mut_parts(|vec, old_mapping| {
                    old_mapping.shape().with_dims(|old_dims| {
                        if new_len == 0 {
                            vec.clear();
                        } else if new_dims[1..] == old_dims[1..] {
                            vec.resize_with(new_len, &mut f);
                        } else {
                            #[cfg(not(feature = "nightly"))]
                            let mut new_vec = Vec::with_capacity(new_len);
                            #[cfg(feature = "nightly")]
                            let mut new_vec =
                                Vec::with_capacity_in(new_len, vec.allocator().clone());

                            copy_dim::<T, S, A>(
                                &mut DropGuard::new(vec),
                                &mut new_vec,
                                old_dims,
                                new_dims,
                                &mut f,
                            );

                            *vec = new_vec;
                        }
                    });

                    old_mapping.shape_mut().with_mut_dims(|dims| dims.copy_from_slice(new_dims));
                });
            }
        }
    }

    pub(crate) unsafe fn set_mapping(&mut self, new_mapping: DenseMapping<S>) {
        debug_assert!(new_mapping.shape().checked_len().is_some(), "invalid length");
        debug_assert!(new_mapping.len() <= self.capacity, "length exceeds capacity");

        *self.slice.mapping_mut() = new_mapping;
    }

    #[cfg(not(feature = "nightly"))]
    pub(crate) unsafe fn with_mut_parts<U, F>(&mut self, f: F) -> U
    where
        F: FnOnce(&mut Vec<T>, &mut DenseMapping<S>) -> U,
    {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            tensor: &'a mut RawTensor<T, S, A>,
            vec: ManuallyDrop<Vec<T>>,
        }

        impl<T, S: Shape, A: Allocator> Drop for DropGuard<'_, T, S, A> {
            fn drop(&mut self) {
                unsafe {
                    self.tensor.slice.set_ptr(self.vec.as_mut_ptr());
                    self.tensor.capacity = self.vec.capacity();

                    // Cleanup in case of length mismatch (e.g. due to allocation failure)
                    if self.vec.len() != self.tensor.slice.mapping().len() {
                        *self.tensor.slice.mapping_mut() = DenseMapping::default();
                        ptr::drop_in_place(self.vec.as_mut_slice());
                    }
                }
            }
        }

        let vec =
            Vec::from_raw_parts(self.slice.as_mut_ptr(), self.slice.mapping().len(), self.capacity);

        let mut guard = DropGuard { tensor: self, vec: ManuallyDrop::new(vec) };

        let mapping = guard.tensor.slice.mapping_mut();
        let result = f(&mut guard.vec, mapping);

        debug_assert!(Some(guard.vec.len()) == mapping.shape().checked_len(), "length mismatch");

        guard.tensor.slice.set_ptr(guard.vec.as_mut_ptr());
        guard.tensor.capacity = guard.vec.capacity();

        mem::forget(guard);

        result
    }

    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn with_mut_parts<U, F>(&mut self, f: F) -> U
    where
        F: FnOnce(&mut Vec<T, A>, &mut DenseMapping<S>) -> U,
    {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            tensor: &'a mut RawTensor<T, S, A>,
            vec: ManuallyDrop<Vec<T, A>>,
        }

        impl<T, S: Shape, A: Allocator> Drop for DropGuard<'_, T, S, A> {
            fn drop(&mut self) {
                unsafe {
                    self.tensor.slice.set_ptr(self.vec.as_mut_ptr());
                    self.tensor.capacity = self.vec.capacity();
                    self.tensor.alloc = ManuallyDrop::new(ptr::read(self.vec.allocator()));

                    // Cleanup in case of length mismatch (e.g. due to allocation failure)
                    if self.vec.len() != self.tensor.slice.mapping().len() {
                        *self.tensor.slice.mapping_mut() = DenseMapping::default();
                        ptr::drop_in_place(self.vec.as_mut_slice());
                    }
                }
            }
        }

        let vec = unsafe {
            Vec::from_raw_parts_in(
                self.slice.as_mut_ptr(),
                self.slice.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };

        let mut guard = DropGuard { tensor: self, vec: ManuallyDrop::new(vec) };

        let mapping = unsafe { guard.tensor.slice.mapping_mut() };
        let result = f(&mut guard.vec, mapping);

        debug_assert!(Some(guard.vec.len()) == mapping.shape().checked_len(), "length mismatch");

        guard.tensor.slice.set_ptr(guard.vec.as_mut_ptr());
        guard.tensor.capacity = guard.vec.capacity();
        guard.tensor.alloc = ManuallyDrop::new(ptr::read(guard.vec.allocator()));

        mem::forget(guard);

        result
    }

    pub(crate) fn with_vec<U, F: FnOnce(&vec_t!(T, A)) -> U>(&self, f: F) -> U {
        #[cfg(not(feature = "nightly"))]
        let vec = unsafe {
            Vec::from_raw_parts(
                self.slice.as_ptr() as *mut T,
                self.slice.mapping().len(),
                self.capacity,
            )
        };
        #[cfg(feature = "nightly")]
        let vec = unsafe {
            Vec::from_raw_parts_in(
                self.slice.as_ptr() as *mut T,
                self.slice.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };

        f(&ManuallyDrop::new(vec))
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for RawTensor<T, S, A> {
    fn clone(&self) -> Self {
        unsafe { Self::from_parts(self.with_vec(|vec| vec.clone()), self.slice.mapping().clone()) }
    }

    fn clone_from(&mut self, source: &Self) {
        unsafe {
            self.with_mut_parts(|dst, mapping| {
                source.with_vec(|src| dst.clone_from(src));
                mapping.clone_from(source.slice.mapping());
            });
        }
    }
}

impl<T, S: Shape, A: Allocator> Drop for RawTensor<T, S, A> {
    #[cfg(not(feature = "nightly"))]
    fn drop(&mut self) {
        _ = unsafe {
            Vec::from_raw_parts(self.slice.as_mut_ptr(), self.slice.mapping().len(), self.capacity)
        };
    }

    #[cfg(feature = "nightly")]
    fn drop(&mut self) {
        _ = unsafe {
            Vec::from_raw_parts_in(
                self.slice.as_mut_ptr(),
                self.slice.mapping().len(),
                self.capacity,
                ptr::read(&*self.alloc),
            )
        };
    }
}

unsafe impl<T: Send, S: Shape, A: Allocator + Send> Send for RawTensor<T, S, A> {}
unsafe impl<T: Sync, S: Shape, A: Allocator + Sync> Sync for RawTensor<T, S, A> {}

impl<'a, T, A: Allocator> DropGuard<'a, T, A> {
    fn new(vec: &'a mut vec_t!(T, A)) -> Self {
        let len = vec.len();

        unsafe {
            vec.set_len(0);
        }

        Self { ptr: vec.as_mut_ptr(), len, phantom: PhantomData }
    }
}

impl<T, A: Allocator> Drop for DropGuard<'_, T, A> {
    fn drop(&mut self) {
        unsafe {
            ptr::slice_from_raw_parts_mut(self.ptr, self.len).drop_in_place();
        }
    }
}

unsafe fn copy_dim<T, S: Shape, A: Allocator>(
    old_vec: &mut DropGuard<T, A>,
    new_vec: &mut vec_t!(T, A),
    old_dims: &[usize],
    new_dims: &[usize],
    f: &mut impl FnMut() -> T,
) {
    let old_stride: usize = old_dims[1..].iter().product();
    let new_stride: usize = new_dims[1..].iter().product();

    let old_size = old_dims[0];
    let new_size = new_dims[0];

    let min_size = old_size.min(new_size);

    if old_dims.len() > 1 {
        // Avoid very long compile times for release build with MIR inlining,
        // by avoiding recursion until types are known.
        //
        // This is a workaround until const if is available, see #3582 and #122301.

        unsafe fn dummy<T, A: Allocator>(
            _: &mut DropGuard<T, A>,
            _: &mut vec_t!(T, A),
            _: &[usize],
            _: &[usize],
            _: &mut impl FnMut() -> T,
        ) {
            unreachable!();
        }

        let g = const {
            match S::RANK {
                Some(..2) => dummy::<T, A>,
                _ => copy_dim::<T, S::Tail, A>,
            }
        };

        for _ in 0..min_size {
            g(old_vec, new_vec, &old_dims[1..], &new_dims[1..], f);
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
