/// Order for indexing and iteration over array elements.
pub trait Order: Copy {
    /// Returns the first or second input parameter depending on the element order.
    fn select<T>(cm: T, rm: T) -> T;
}

/// Column-major order, for indexing and iteration over array elements.
#[derive(Clone, Copy)]
pub struct ColumnMajor;

/// Row-major order, for indexing and iteration over array elements.
#[derive(Clone, Copy)]
pub struct RowMajor;

impl Order for ColumnMajor {
    fn select<T>(cm: T, _: T) -> T {
        cm
    }
}

impl Order for RowMajor {
    fn select<T>(_: T, rm: T) -> T {
        rm
    }
}
