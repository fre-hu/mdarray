/// Order for indexing and iteration over array elements.
pub trait Order {
    /// True if the array has column-major element order.
    const IS_COLUMN_MAJOR: bool;

    /// True if the array has row-major element order.
    const IS_ROW_MAJOR: bool;

    /// Returns the first or second input parameter depending on the element order.
    #[must_use]
    fn select<T>(cm: T, rm: T) -> T;
}

/// Column-major order, for indexing and iteration over array elements.
pub struct ColumnMajor;

/// Row-major order, for indexing and iteration over array elements.
pub struct RowMajor;

impl Order for ColumnMajor {
    const IS_COLUMN_MAJOR: bool = true;
    const IS_ROW_MAJOR: bool = false;

    fn select<T>(cm: T, _: T) -> T {
        cm
    }
}

impl Order for RowMajor {
    const IS_COLUMN_MAJOR: bool = false;
    const IS_ROW_MAJOR: bool = true;

    fn select<T>(_: T, rm: T) -> T {
        rm
    }
}
