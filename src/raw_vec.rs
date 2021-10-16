use std::alloc::{self, Allocator, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::{cmp, mem};

// Maximum SIMD vector size
const MIN_ALIGN: usize = if cfg!(target_feature = "avx512f") {
    64
} else if cfg!(target_feature = "avx") {
    32
} else {
    16
};

pub struct RawVec<T, A: Allocator> {
    ptr: NonNull<T>,
    capacity: usize,
    alloc: A,
    _marker: PhantomData<T>,
}

impl<T, A: Allocator> RawVec<T, A> {
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub unsafe fn from_raw_parts_in(ptr: *mut T, capacity: usize, alloc: A) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            capacity,
            alloc,
            _marker: PhantomData,
        }
    }

    pub fn grow(&mut self, capacity: usize) {
        self.grow_exact(cmp::max(2 * self.capacity, capacity));
    }

    pub fn grow_exact(&mut self, capacity: usize) {
        let new_layout = Self::layout(capacity.checked_mul(mem::size_of::<T>()).unwrap());

        let result = if self.capacity == 0 {
            self.alloc.allocate(new_layout)
        } else {
            let old_layout = Self::layout(self.capacity * mem::size_of::<T>());

            unsafe { self.alloc.grow(self.ptr.cast(), old_layout, new_layout) }
        };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(_) => alloc::handle_alloc_error(new_layout),
        };

        self.ptr = ptr.cast();
        self.capacity = ptr.len() / mem::size_of::<T>();
    }

    pub fn new_in(alloc: A) -> Self {
        Self {
            ptr: NonNull::dangling(),
            capacity: 0,
            alloc,
            _marker: PhantomData,
        }
    }

    pub fn shrink(&mut self, capacity: usize) {
        let old_layout = Self::layout(self.capacity * mem::size_of::<T>());
        let new_layout = Self::layout(capacity * mem::size_of::<T>());

        let result = unsafe { self.alloc.shrink(self.ptr.cast(), old_layout, new_layout) };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(_) => alloc::handle_alloc_error(new_layout),
        };

        self.ptr = ptr.cast();
        self.capacity = ptr.len() / mem::size_of::<T>();
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let layout = Self::layout(capacity.checked_mul(mem::size_of::<T>()).unwrap());

        let ptr = match alloc.allocate(layout) {
            Ok(ptr) => ptr,
            Err(_) => alloc::handle_alloc_error(layout),
        };

        Self {
            ptr: ptr.cast(),
            capacity: ptr.len() / mem::size_of::<T>(),
            alloc,
            _marker: PhantomData,
        }
    }

    fn layout(size: usize) -> Layout {
        assert!(usize::BITS == 64 || size <= isize::MAX as usize);

        let align = cmp::max(mem::align_of::<T>(), MIN_ALIGN);

        unsafe { Layout::from_size_align_unchecked(size, align) }
    }
}

impl<T, A: Allocator> Drop for RawVec<T, A> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Self::layout(self.capacity * mem::size_of::<T>());

            unsafe { self.alloc.deallocate(self.ptr.cast(), layout) }
        }
    }
}

unsafe impl<T: Send, A: Allocator> Send for RawVec<T, A> {}
unsafe impl<T: Sync, A: Allocator> Sync for RawVec<T, A> {}
