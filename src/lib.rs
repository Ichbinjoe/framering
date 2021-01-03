/*
 *  Copyright (C) 2020  Joe Hirschfeld <j@ibj.io>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! A 'deque of deques' all sharing the same contiguous memory space. `FramedRing` enables creation
//! of multiple deques without needing to allocate a separate heap space for each sequence. As the
//! name implies, `FramedRing` operates similarly to a ring buffer - items in all frames are stored
//! in the same buffer where deallocated frames on the tail of the buffer free up capacity for new
//! element pushes on the head frame.
//!
//! Frame creation and element appending likely will not incur a memory allocation unless the ring
//! is full. Operations such as creating a new frame, appending to a frame, random access into a
//! frame, and iterating over a frame are O(1). Operations which may incur a memory allocation such
//! as frame creation or element appending will only incur a O(n) penalty after memory allocation
//! for expansion.
//!
//! This implementation has flexible capacity behaviors via custom implementations of Capacity - if
//! an application requires unique capacity behaviors, one can be provided.
//!
//! # Examples
//!
//! Rings (the backing store for frames) can be created using the new method - this creates an
//! empty ring. Memory isn't allocated until the first frame is created.
//! ```
//! # use ::framering::*;
//! let ring: FramedRing<i32, Pow2Capacity> = FramedRing::new();
//! ```
//!
//! An initial seed frame can be created using the frame method and have items appended to it...
//! ```
//! # use ::framering::*;
//! let mut ring = FramedRing::<i32, Pow2Capacity>::new();
//! let mut frame = ring.frame();
//! frame.push(1);
//! frame.push(2);
//! frame.push(3);
//! ```
//!
//! ... and other frames can be created after with more elements.
//! ```
//! # use ::framering::*;
//! let mut ring = FramedRing::<i32, LinearCapacity>::new();
//!
//! let mut frame1 = ring.frame();
//! frame1.push(1);
//!
//! let (frame1_ro, mut frame2) = frame1.next();
//! frame2.push(2);
//!
//! let (frame2_ro, mut frame3) = frame2.next();
//! frame3.push(3);
//! ```
//!
//!
//! A FramedRing may have multiple frames at once, but only one (the head frame) may have items
//! appended to it. Frames (including the head) can be dropped at any time and all objects stored
//! in that frame will be dropped as well. Space is reclaimed when the tail frame is dropped - when
//! that happens, the oldest non-dropped frame will become the new tail and space before then will
//! be reclaimed.
//!
//! Elements within frames are indexable (with a complexity of O(1)), and appends to the head frame
//! are also O(1) assuming the ring has space (otherwise there is a complexity of O(n)).
//!

#![feature(untagged_unions)]

use std::alloc;
use std::cell::Cell;
use std::mem::{align_of, ManuallyDrop, MaybeUninit};

/// A header inserted into the FramedRing at the head of each frame
#[derive(Copy, Clone)]
struct FrameHeader {
    /// The index of the next frame header
    next: usize,
    /// The index of the last frame header (or the index of this frame header, if this is the tail)
    last: usize,
}

// Note: Unions never call their destructors as it is impossible to tell which element is actually
// initialized within the union. When a frame is destructed, all elements within the frame also
// need dropped.
/// An entry within the FramedRing
union RingElement<T> {
    /// A frame header which is placed at the beginning of each Frame as a marker to the end of the
    /// frame as well as a running reference count
    header: FrameHeader,
    /// A frame element
    element: ManuallyDrop<T>,
}

/// Defines the capacity behavior of a FramedRing.
pub trait Capacity: Clone + Copy + Default {
    type Baser;
    type Masker;

    /// Current size in number of elements of this Capacity.
    fn size(&self) -> usize;
    /// Current size in bytes of this Capacity for the given type.
    fn size_array<T>(&self) -> usize;
    /// Returns a capacity which fits at least the passed capacity.
    fn increase(&self, new_capacity: usize) -> Result<Self, ()>;
    /// Returns a mask object to be used with mask(mask, v) to generate masks for some value v
    fn make_mask(&self) -> Self::Masker;
    /// Returns a base object to be used with base(base, v) to generate bases for some value v
    fn make_base(&self) -> Self::Baser;
    /// Masks a given value v with the given masker (generated with this Capacity)
    fn mask(mask: &Self::Masker, v: usize) -> usize;
    /// Bases a given value v with the given masker (generated with this Capacity)
    fn base(mask: &Self::Baser, v: usize) -> usize;
    /// Adds a relative index to an absolute index in a way which is compatible with the capacity
    /// scheme
    fn add(a: usize, b: usize) -> usize;
}

/// Implementation of capacity which represents the capacity as a power of 2. Supports infinite
/// addressing (will not eventually result in a panic, no matter how many elements are added over
/// the lifetime of the FramedRing).
///
/// # Example
///
/// ```
/// # use ::framering::*;
/// let mut ring = FramedRing::<i32, Pow2Capacity>::with_capacity(Pow2Capacity::Pow2(4));
/// assert_eq!(ring.capacity(), 16);
/// ```
#[derive(Clone, Copy)]
pub enum Pow2Capacity {
    Zero,
    Pow2(usize),
}

impl Default for Pow2Capacity {
    fn default() -> Self {
        Pow2Capacity::Zero
    }
}

impl Capacity for Pow2Capacity {
    type Baser = usize;
    type Masker = usize;

    fn size(&self) -> usize {
        match self {
            Pow2Capacity::Zero => 0,
            Pow2Capacity::Pow2(v) => 1 << v,
        }
    }

    fn size_array<T>(&self) -> usize {
        match self {
            Pow2Capacity::Zero => 0,
            Pow2Capacity::Pow2(v) => std::mem::size_of::<RingElement<T>>() << v,
        }
    }

    fn increase(&self, new_capacity: usize) -> Result<Self, ()> {
        if new_capacity == 0 {
            Ok(Pow2Capacity::Zero)
        } else {
            let bits = std::mem::size_of::<usize>() * 8;
            let shift = bits as usize - new_capacity.leading_zeros() as usize;
            // Minimum of 16 elements (2 ^ 4)
            Ok(Pow2Capacity::Pow2(std::cmp::max(shift, 4)))
        }
    }

    fn make_mask(&self) -> usize {
        match self {
            Pow2Capacity::Zero => 0,
            Pow2Capacity::Pow2(v) => (1 << v) - 1,
        }
    }

    fn make_base(&self) -> usize {
        !self.make_mask()
    }

    fn mask(mask: &usize, val: usize) -> usize {
        val & mask
    }

    fn base(mask: &usize, val: usize) -> usize {
        val & mask
    }

    fn add(a: usize, b: usize) -> usize {
        a.wrapping_add(b)
    }
}

/// Implementation of capacity which represents the capacity as usize. Does not support infinite
/// addressing, and will panic if more than usize::MAX elements are inserted in a forward fashion
/// (due to addition overflow).
///
/// LinearCapacity does not necessarily grow in a doubling fashion - if bulk addition requires more
/// elements than a standard doubling, the FramedRing will expand to fit all of the elements
/// instead.
///
/// # Example
///
/// ```
/// # use ::framering::*;
/// let mut ring = FramedRing::<i32, LinearCapacity>::with_capacity(LinearCapacity{cap: 9});
/// assert_eq!(ring.capacity(), 9);
/// ```
#[derive(Clone, Copy)]
pub struct LinearCapacity {
    pub cap: usize,
}

impl Default for LinearCapacity {
    fn default() -> Self {
        LinearCapacity { cap: 0 }
    }
}

impl Capacity for LinearCapacity {
    type Baser = usize;
    type Masker = usize;

    fn size(&self) -> usize {
        self.cap
    }

    fn size_array<T>(&self) -> usize {
        self.cap * std::mem::size_of::<RingElement<T>>()
    }

    fn increase(&self, new_capacity: usize) -> Result<Self, ()> {
        let default_new_size = if new_capacity == 0 { 8 } else { self.cap * 2 };

        // At least double, but if the increase is over this just allocate the exact amount
        Ok(LinearCapacity {
            cap: std::cmp::max(default_new_size, new_capacity),
        })
    }

    fn make_mask(&self) -> usize {
        self.cap
    }

    fn make_base(&self) -> usize {
        self.cap
    }

    fn mask(mask: &usize, val: usize) -> usize {
        val % mask
    }

    fn base(mask: &usize, val: usize) -> usize {
        val / mask
    }

    fn add(a: usize, b: usize) -> usize {
        a + b
    }
}

/// A ring buffer which exposes operations via 'frames'.
pub struct FramedRing<T, Cap: Capacity> {
    /// Pointer to the memory range containing all of the elements
    ring: Cell<*mut MaybeUninit<RingElement<T>>>,
    /// The base of the ring, which contains the index of the root FrameHeader
    base: Cell<usize>,
    /// The head of the ring, which will contain the next element to be inserted
    head: Cell<usize>,
    /// Capacity of the ring
    capacity: Cell<Cap>,
}

fn create_buffer<T, Cap: Capacity>(c: Cap) -> *mut std::mem::MaybeUninit<RingElement<T>> {
    let size = c.size_array::<T>();
    if size == 0 {
        std::ptr::null_mut()
    } else {
        unsafe {
            let new_buffer_layout =
                alloc::Layout::from_size_align_unchecked(size, align_of::<RingElement<T>>());

            std::mem::transmute(alloc::alloc(new_buffer_layout))
        }
    }
}

macro_rules! floored_min {
    ($floor:expr) => { $floor };
    ($floor:expr, $a:expr) => {std::cmp::max($floor, $a)};
    ($floor:expr, $a:expr, $b:expr) => {
        floored_min!($floor, std::cmp::min($a, $b))
    };
    ($floor:expr, $a:expr, $b:expr, $($extra:expr), +) => {
        floored_min!($floor, std::cmp::min($a, $b), $($extra),+)
    };
}

unsafe fn copy_range<T, Cap: Capacity>(
    from_buf: *const MaybeUninit<RingElement<T>>,
    to_buf: *mut MaybeUninit<RingElement<T>>,
    from_mask: &Cap::Masker,
    to_mask: &Cap::Masker,
    start_index: usize,
    end_index: usize,
) {
    let base_from = Cap::mask(from_mask, start_index);
    let base_to = Cap::mask(to_mask, start_index);
    std::ptr::copy_nonoverlapping(
        from_buf.add(base_from),
        to_buf.add(base_to),
        end_index - start_index,
    );
}

impl<T, Cap: Capacity> FramedRing<T, Cap> {
    /// Creates a new FramedRing with the default Cap size
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<(), LinearCapacity>::new();
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(Cap::default())
    }

    /// Creates a new FramedRing with the given capacity
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::with_capacity(LinearCapacity{cap: 32});
    /// assert_eq!(ring.capacity(), 32);
    /// ```
    pub fn with_capacity(c: Cap) -> Self {
        let buffer = create_buffer(c);
        FramedRing {
            ring: Cell::new(buffer),
            base: Cell::new(0),
            head: Cell::new(0),
            capacity: Cell::new(c),
        }
    }

    /// Creates a new root frame. Panics if this is called when another frame is still in scope
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// ```
    pub fn frame<'ring>(&'ring self) -> RingFrameMut<'ring, T, Cap> {
        self.frame_reserve(0)
    }

    /// Creates a new root frame with the given capacity for inserting elements. Panics if this is
    /// called when another frame is still in scope
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame_reserve(4);
    /// assert_eq!(ring.capacity(), 5);
    /// frame.push(1);
    /// ```
    pub fn frame_reserve<'ring>(&'ring self, frame_capacity: usize) -> RingFrameMut<'ring, T, Cap> {
        let start = self.head.get();
        if start != self.base.get() {
            panic!("attempting to start a new root frame with already-modifiable frames are using this ring");
        }

        let mask = if self.ring.get().is_null() {
            // Ring must be created, but if we create the ring (assuming the Cap implementation is
            // correct) then the buffer will be non-zero (and thus won't require another capacity
            // check).
            unsafe {
                self.set_blank_buffer(frame_capacity + 1)
                    .expect("unable to allocate buffer");
                self.capacity.get().make_mask()
            }
        } else {
            // Otherwise, we aren't entirely sure we can fit this new frame, so perform a capacity
            // check first.
            unsafe {
                self.enforce_fit(frame_capacity + 1)
                    .expect("unable to expand buffer")
            }
        };

        unsafe {
            self.append_to_ring(
                &mask,
                RingElement {
                    header: FrameHeader {
                        next: start + 1,
                        last: start,
                    },
                },
            );
        }

        RingFrameMut {
            f: RingFrame {
                ring: self,
                start,
                live_at: start + 1,
                live_to: start + 1,
            },
        }
    }

    unsafe fn set_blank_buffer(&self, expected_size: usize) -> Result<(), ()> {
        // If we've hit this, self.ring == nullptr, and self.capacity _should_ give size() == 0 (if
        // it doesn't, that's a bug). Increment the capacity here and blind write the new buffer.
        let c = self.capacity.get().increase(expected_size)?;
        self.ring.set(create_buffer(c));
        self.capacity.set(c);
        Ok(())
    }

    unsafe fn append_to_ring(&self, mask: &Cap::Masker, re: RingElement<T>) {
        let old_head = self.head.get();
        self.head.set(old_head + 1);

        let masked = Cap::mask(&mask, old_head);

        let ptr = self.ring.get().add(masked);
        ptr.write(MaybeUninit::new(re));
    }

    unsafe fn enforce_fit(&self, new_capacity: usize) -> Result<Cap::Masker, ()> {
        let current_capacity = self.capacity.get();
        let current_size = current_capacity.size();
        let current_mask = current_capacity.make_mask();

        if new_capacity > current_size {
            // We have to expand to fit the new elements
            let new_capacity = current_capacity.increase(new_capacity)?;
            let new_mask = new_capacity.make_mask();
            let new_buffer = create_buffer(new_capacity);

            // The elements are copied over in (up to) three shots, but generally the naive cost of
            // doing these copies will always be the same (we only copy parts of the buffer that we
            // are using) but generally these copies will be small enough that the real bottleneck
            // for the system will likely be branching or some other code flow problems. This
            // should be called infrequently enough (at least in the use I want it for :) ) that
            // its not worth optimizing super heavily (for example in the special pow2 case, where
            // this can be simplified to be a two part pivot).

            let old_baser = self.capacity.get().make_base();
            let new_baser = new_capacity.make_base();

            let base = self.base.get();
            let head = self.head.get();
            let old_pivot = Cap::base(&old_baser, head);
            let new_pivot = Cap::base(&new_baser, head);

            let from_buffer = self.ring.get();

            let pivot1 = floored_min!(base, head, old_pivot, new_pivot);
            copy_range::<T, Cap>(
                from_buffer,
                new_buffer,
                &current_mask,
                &new_mask,
                base,
                pivot1,
            );
            if pivot1 != head {
                let pivot2 = floored_min!(pivot1, head, old_pivot, new_pivot);
                copy_range::<T, Cap>(
                    from_buffer,
                    new_buffer,
                    &current_mask,
                    &new_mask,
                    pivot1,
                    pivot2,
                );
                if pivot2 != head {
                    let pivot3 = head;
                    copy_range::<T, Cap>(
                        from_buffer,
                        new_buffer,
                        &current_mask,
                        &new_mask,
                        pivot2,
                        pivot3,
                    );
                }
            }

            // replace the current buffer incase dealloc panics below
            self.ring.set(new_buffer);
            self.capacity.set(new_capacity);

            let old_buffer_layout = alloc::Layout::from_size_align_unchecked(
                current_capacity.size_array::<T>(),
                align_of::<RingElement<T>>(),
            );

            // Deallocate the old buffer
            std::alloc::dealloc(std::mem::transmute(from_buffer), old_buffer_layout);

            Ok(new_mask)
        } else {
            // already an appropriate size
            Ok(current_mask)
        }
    }

    /// Tries to promote a given frame to a mutable frame which can be appended to. Returns Ok with
    /// the new mutable frame if the passed frame could be promoted or returns Err with the passed
    /// frame if unable.
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// let (frame_ro, frame2) = frame.next();
    /// let (frame2_ro, frame3) = frame2.next();
    /// drop(frame3);
    /// assert!(ring.try_promote(frame_ro).is_err());
    /// assert!(ring.try_promote(frame2_ro).is_ok());
    /// ```
    pub fn try_promote<'ring>(
        &'ring self,
        frame: RingFrame<'ring, T, Cap>,
    ) -> Result<RingFrameMut<'ring, T, Cap>, RingFrame<'ring, T, Cap>> {
        unsafe {
            let mut header = &mut self.get_mut(frame.start).header;
            if header.next != self.head.get() {
                Err(frame)
            } else {
                if self.head.get() != frame.live_to {
                    // the head and header.next need set back. Why? Otherwise, we are throwing away
                    // capacity and some of the other code below assumes that when on the head,
                    // live_to == head == header.next.
                    //
                    // This happens if stupid things happen.
                    self.head.set(frame.live_to);
                    header.next = frame.live_to;
                }

                let r = Ok(RingFrameMut {
                    f: RingFrame {
                        ring: self,
                        start: frame.start,
                        live_at: frame.live_at,
                        live_to: frame.live_to,
                    },
                });

                std::mem::forget(frame);

                return r;
            }
        }
    }

    /// Promotes a given frame, or panics if passed frame is not able to be promoted
    ///
    /// # Errors
    ///
    /// Panics if the passed frame ring is unable to be promoted for whatever reason.
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// let (frame_ro, frame2) = frame.next();
    /// drop(frame2);
    /// ring.promote(frame_ro);
    /// ```
    pub fn promote<'ring>(
        &'ring self,
        frame: RingFrame<'ring, T, Cap>,
    ) -> RingFrameMut<'ring, T, Cap> {
        match self.try_promote(frame) {
            Ok(f) => f,
            Err(_) => panic!("frame unable to be promoted"),
        }
    }

    /// Returns the current mask of the container
    fn mask(&self, index: usize) -> usize {
        let mask = self.capacity.get().make_mask();
        Cap::mask(&mask, index)
    }

    /// Gets a reference to the ring element at a unmasked index. Unsafe since an uninitialized
    /// element may be returned.
    unsafe fn get<'a>(&'a self, i: usize) -> &'a RingElement<T> {
        self.get_masked(self.mask(i))
    }

    /// Gets a mutable reference to the ring element at a unmasked index. Unsafe since an
    /// uninitialized element may be returned.
    unsafe fn get_mut<'a>(&'a self, i: usize) -> &'a mut RingElement<T> {
        self.get_masked_mut(self.mask(i))
    }

    /// Gets a reference to the ring element at a masked index. Unsafe since
    /// an uninitialized element may be returned, and an unmaked element will index out of the
    /// allocated memory range.
    unsafe fn get_masked<'a>(&'a self, i: usize) -> &'a RingElement<T> {
        std::mem::transmute(&*self.ring.get().add(i))
    }

    /// Gets a mutable reference to the ring element at a masked index. Unsafe since an
    /// uninitialized element may be returned, and an unmaked element will index out of the
    /// allocated memory range.
    unsafe fn get_masked_mut<'a>(&'a self, i: usize) -> &'a mut RingElement<T> {
        std::mem::transmute(&mut *self.ring.get().add(i))
    }

    /// Returns the current capacity of the entire ring
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::with_capacity(LinearCapacity{cap: 32});
    /// assert_eq!(ring.capacity(), 32);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity.get().size()
    }

    /// Returns the used capacity of the entire ring. Counts headers as well as elements which may
    /// have been dropped but not free to be used again yet.
    ///
    /// About the only use for this is for understanding used capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// // Header + element
    /// assert_eq!(ring.size(), 2);
    /// ```
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// let (frame_ro, mut frame2) = frame.next();
    /// frame2.push(1);
    /// let (frame2_ro, mut frame3) = frame2.next();
    /// frame3.push(1);
    /// assert_eq!(ring.size(), 6);
    /// drop(frame2_ro);
    /// assert_eq!(ring.size(), 6);
    /// drop(frame_ro);
    /// assert_eq!(ring.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.head.get() - self.base.get()
    }
}

impl<T, Cap: Capacity> Drop for FramedRing<T, Cap> {
    fn drop(&mut self) {
        // If we can drop, that means all child frames have been dropped and we just need to free
        // the buffer.

        unsafe {
            let layout = alloc::Layout::from_size_align_unchecked(
                self.capacity.get().size_array::<RingElement<T>>(),
                align_of::<RingElement<T>>(),
            );

            alloc::dealloc(std::mem::transmute(self.ring.get()), layout);
        }
    }
}

/// A mutable frame which allows elements to be appended to the end
pub struct RingFrameMut<'ring, T, Cap: Capacity> {
    f: RingFrame<'ring, T, Cap>,
}

/// An immutable frame which does not allow elements to be appended
pub struct RingFrame<'ring, T, Cap: Capacity> {
    ring: &'ring FramedRing<T, Cap>,
    start: usize,
    live_at: usize,
    live_to: usize,
}

impl<'ring, T, Cap: Capacity> Drop for RingFrame<'ring, T, Cap> {
    fn drop(&mut self) {
        // Drop all of our contents, then attempt to progress base as far as we can.  We can
        // progress this all the way up until base == head, in which case this was the last frame
        // in the ring (not that it matters to us, but interesting to know).

        unsafe {
            // This mask is only valid over while the ring doesn't change size.
            let mask = self.ring.capacity.get().make_mask();
            let header = &mut self
                .ring
                .get_masked_mut(Cap::mask(&mask, self.start))
                .header;
            if std::mem::needs_drop::<T>() {
                for i in self.live_at..self.live_to {
                    ManuallyDrop::drop(&mut self.ring.get_masked_mut(Cap::mask(&mask, i)).element);
                }
            }

            if header.last != self.start {
                // We have a previous header somewhere and are not the base
                if header.next == self.ring.head.get() {
                    // But we are the head. So roll the head back to our index
                    self.ring.head.set(self.start);
                } else {
                    // Otherwise we have another header beyond us, and our previous needs
                    // rewritten
                    let mut prev_header = &mut self
                        .ring
                        .get_masked_mut(Cap::mask(&mask, header.last))
                        .header;
                    prev_header.next = header.next;

                    // Our next also needs rewritten
                    let mut next_header = &mut self
                        .ring
                        .get_masked_mut(Cap::mask(&mask, header.next))
                        .header;
                    next_header.last = header.last;
                }
            } else {
                // We are the base - rewrite the base to our next (which could be to head - meaning
                // the ring is empty)
                self.ring.base.set(header.next);
                if header.next != self.ring.head.get() {
                    // Next is another frame - rewrite their last to be whatever our last was.
                    let mut next_header = &mut self
                        .ring
                        .get_masked_mut(Cap::mask(&mask, header.next))
                        .header;
                    next_header.last = header.last;
                }
            }
        }
    }
}

/// Iterator over elements within a RingFrame
pub struct RingFrameIter<'a, T, Cap: Capacity> {
    ring: &'a FramedRing<T, Cap>,
    i: usize,
    end: usize,
}

impl<'a, T, Cap: Capacity> Iterator for RingFrameIter<'a, T, Cap> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.end {
            None
        } else {
            unsafe {
                let item = &self.ring.get(self.i).element;
                self.i += 1;
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.i, Some(self.end - self.i))
    }
}

/// Iterator over elements within a RingFrame
pub struct RingFrameIntoIter<'a, T, Cap: Capacity> {
    f: RingFrame<'a, T, Cap>,
    end: usize,
}

impl<'a, T, Cap: Capacity> Iterator for RingFrameIntoIter<'a, T, Cap> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.f.live_at >= self.end {
            None
        } else {
            unsafe {
                // This is basically what happens in self.f.ring.get(), but without doing memory
                // transmutation because we actually don't want to do it here.
                let element = (&*self.f.ring.ring.get().add(self.f.live_at))
                    .as_ptr()
                    .read()
                    .element;
                self.f.live_at += 1;
                Some(ManuallyDrop::into_inner(element))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.f.live_at, Some(self.end - self.f.live_at))
    }
}

impl<'ring, T, Cap: Capacity> std::convert::From<RingFrameMut<'ring, T, Cap>>
    for RingFrame<'ring, T, Cap>
{
    fn from(rfm: RingFrameMut<'ring, T, Cap>) -> Self {
        rfm.f
    }
}

impl<'ring, T, Cap: Capacity> RingFrame<'ring, T, Cap> {
    fn header<'a>(&'a self) -> &'a mut FrameHeader {
        unsafe { &mut self.ring.get_mut(self.start).header }
    }

    /// Length in elements of the frame
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// frame.push(2);
    /// frame.push(3);
    /// assert_eq!(frame.as_ref().len(), 3);
    /// ```
    ///
    pub fn len(&self) -> usize {
        self.live_to - self.live_at
    }

    /// Gets the given item in the frame as a reference, or None if out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// let frame_ro = frame.downgrade();
    /// assert_eq!(*frame_ro.get(0).unwrap(), 1);
    /// assert_eq!(frame_ro.get(1), None);
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        let i = self.live_at + index;
        if i >= self.live_to {
            None
        } else {
            Some(unsafe { &self.ring.get(i).element })
        }
    }

    /// Gets the given item in the frame as a mutable reference, or None if out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// let mut frame_ro = frame.downgrade();
    /// assert_eq!(*frame_ro.get(0).unwrap(), 1);
    /// *frame_ro.get_mut(0).unwrap() = 2;
    /// assert_eq!(*frame_ro.get(0).unwrap(), 2);
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let i = self.live_at + index;
        if i >= self.live_to {
            None
        } else {
            Some(unsafe { &mut self.ring.get_mut(i).element })
        }
    }

    /// Gets the given item in the frame as a reference, performing no bounds checking
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// let frame_ro = frame.downgrade();
    /// assert_eq!(unsafe {*frame_ro.get_unchecked(0)}, 1);
    /// ```
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &self.ring.get(self.live_at + index).element
    }

    /// Gets the given item in the frame as a mutable reference, performing no bounds checking
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// let mut frame_ro = frame.downgrade();
    /// assert_eq!(*frame_ro.get(0).unwrap(), 1);
    /// unsafe {*frame_ro.get_unchecked_mut(0) = 2};
    /// assert_eq!(*frame_ro.get(0).unwrap(), 2);
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut self.ring.get_mut(self.live_at + index).element
    }

    /// An iterator over all of the frame elements
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// frame.push(2);
    /// frame.push(3);
    /// assert_eq!(frame.as_ref().iter().copied().collect::<Vec<i32>>(), vec![1, 2, 3]);
    /// ```
    pub fn iter<'a>(&'a self) -> RingFrameIter<'a, T, Cap> {
        RingFrameIter {
            ring: self.ring,
            i: self.live_at,
            end: self.live_to,
        }
    }
}

impl<'ring, T, Cap: Capacity> IntoIterator for RingFrame<'ring, T, Cap> {
    type Item = T;
    type IntoIter = RingFrameIntoIter<'ring, T, Cap>;

    fn into_iter(self) -> Self::IntoIter {
        let live_to = self.live_to;
        RingFrameIntoIter {
            f: self,
            end: live_to,
        }
    }
}

impl<'ring, T, Cap: Capacity> std::convert::AsRef<RingFrame<'ring, T, Cap>>
    for RingFrameMut<'ring, T, Cap>
{
    fn as_ref(&self) -> &RingFrame<'ring, T, Cap> {
        &self.f
    }
}

impl<'ring, T, Cap: Capacity> RingFrameMut<'ring, T, Cap> {
    /// Freezes the current frame and creates a new mutable frame
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let frame = ring.frame();
    /// let (frame_ro, frame2) = frame.next();
    /// ```
    pub fn next(self) -> (RingFrame<'ring, T, Cap>, RingFrameMut<'ring, T, Cap>) {
        self.next_reserve(1)
    }

    /// Freezes the current frame and creates a new mutable frame with the given amount of reserved
    /// elements
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let frame = ring.frame();
    /// let (frame_ro, frame2) = frame.next_reserve(4);
    /// assert_eq!(ring.capacity(), 6);
    /// ```
    pub fn next_reserve(
        self,
        frame_capacity: usize,
    ) -> (RingFrame<'ring, T, Cap>, RingFrameMut<'ring, T, Cap>) {
        // We need to produce a new frame header for the new frame
        let head = self.f.ring.head.get();
        unsafe {
            let mask = self
                .f
                .ring
                .enforce_fit(self.f.ring.size() + frame_capacity + 1)
                .expect("could not expand capacity");
            self.f.ring.append_to_ring(
                &mask,
                RingElement {
                    header: FrameHeader {
                        next: head + 1,
                        last: self.f.start,
                    },
                },
            );
        }

        let ring = self.f.ring;

        (
            self.f,
            RingFrameMut {
                f: RingFrame {
                    ring,
                    start: head,
                    live_at: head + 1,
                    live_to: head + 1,
                },
            },
        )
    }

    /// Pushes an element onto the end of the frame
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.push(1);
    /// ```
    pub fn push(&mut self, element: T) {
        unsafe {
            let mask = self
                .f
                .ring
                .enforce_fit(self.f.ring.size() + 1)
                .expect("could not expand capacity");
            self.f.ring.append_to_ring(
                &mask,
                RingElement {
                    element: ManuallyDrop::new(element),
                },
            );

            self.f.header().next += 1;
            self.f.live_to += 1;
        }
    }

    /// Reserves capacity for at least `additional` more elements into the given frame. The given
    /// capacity may choose to reserve more space.
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, LinearCapacity>::new();
    /// let mut frame = ring.frame();
    /// frame.reserve(4);
    /// assert_eq!(ring.capacity(), 5);
    /// ```
    pub fn reserve(&self, additional: usize) {
        unsafe {
            self.f
                .ring
                .enforce_fit(self.f.ring.size() + additional)
                .expect("could not expand capacity");
        }
    }

    /// Downgrades a RingFrameMut to a RingFrame without producing a successor.
    ///
    /// # Example
    ///
    /// ```
    /// # use ::framering::*;
    /// let mut ring = FramedRing::<i32, Pow2Capacity>::new();
    /// let mut frame = ring.frame();
    /// frame.downgrade();
    /// ```
    pub fn downgrade(self) -> RingFrame<'ring, T, Cap> {
        RingFrame::from(self)
    }
}

impl<'ring, T, Cap: Capacity> Extend<T> for RingFrameMut<'ring, T, Cap> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            let mut it = iter.into_iter();
            let (min_size, _) = it.size_hint();
            let mut size = self.f.ring.size();
            let mut mask = self
                .f
                .ring
                .enforce_fit(size + min_size)
                .expect("could not expand capacity");
            let cap = self.f.ring.capacity();
            let mut cap_left = cap - size;

            while cap_left > 0 {
                if let Some(i) = it.next() {
                    self.f.ring.append_to_ring(
                        &mask,
                        RingElement {
                            element: ManuallyDrop::new(i),
                        },
                    );
                    cap_left -= 1;
                } else {
                    let head = self.f.ring.head.get();
                    self.f.header().next = head;
                    self.f.live_to = head;
                    return;
                }
            }

            size = cap;

            while let Some(i) = it.next() {
                size += 1;
                mask = self
                    .f
                    .ring
                    .enforce_fit(size)
                    .expect("could not expand capacity");
                self.f.ring.append_to_ring(
                    &mask,
                    RingElement {
                        element: ManuallyDrop::new(i),
                    },
                );
            }

            let head = self.f.ring.head.get();
            self.f.header().next = head;
            self.f.live_to = head;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_basic() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let mut frame = ring.frame();
        for i in 0..1024 {
            println!("{}", i);
            frame.push(i);
        }

        let (frame_ro, mut frame2) = frame.next();

        for i in 0..1024 {
            frame2.push(i);
        }

        for i in 0..1024 {
            assert_eq!(*frame_ro.get(i).unwrap(), i as i32);
        }
    }

    #[test]
    fn ring_repromote() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let mut frame = ring.frame();
        for i in 0..512 {
            frame.push(i);
        }

        let (frame_ro, mut frame2) = frame.next();

        for i in 0..1024 {
            frame2.push(i);
        }

        drop(frame2);

        let mut frame3 = ring.promote(frame_ro);

        for i in 512..1024 {
            frame3.push(i);
        }

        for i in 0..1024 {
            assert_eq!(*frame3.as_ref().get(i).unwrap(), i as i32);
        }

        assert_eq!(frame3.as_ref().get(1024), None);
    }

    #[test]
    #[should_panic]
    fn ring_double_frame_no_drop() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let frame = ring.frame();
        let frame2 = ring.frame();
        drop(frame);
        drop(frame2);
    }

    #[test]
    #[should_panic]
    fn ring_bad_promote() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let frame = ring.frame();
        let (frame_ro, frame2) = frame.next();

        ring.promote(frame_ro);
        drop(frame2);
    }

    #[test]
    fn ring_iter() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let mut frame = ring.frame();
        for i in 0..512 {
            frame.push(i);
        }

        let mut itr = frame.as_ref().iter();
        for i in 0..512 {
            assert_eq!(*itr.next().unwrap(), i);
        }
        assert_eq!(itr.next(), None);
    }

    #[test]
    fn ring_into_iter() {
        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let mut frame = ring.frame();
        for i in 0..512 {
            frame.push(i);
        }

        let mut itr = frame.downgrade().into_iter();
        for i in 0..512 {
            assert_eq!(itr.next().unwrap(), i);
        }
        assert_eq!(itr.next(), None);
    }

    struct Dropchecker {
        dropped: *mut bool,
    }

    impl Drop for Dropchecker {
        fn drop(&mut self) {
            unsafe {
                *self.dropped = true;
            }
        }
    }

    #[test]
    fn ring_into_iter_premature_drop() {
        let ring = FramedRing::<Dropchecker, Pow2Capacity>::new();
        let mut frame = ring.frame();
        let mut dropped_a = false;
        let mut dropped_b = false;
        frame.push(Dropchecker {
            dropped: &mut dropped_a,
        });
        frame.push(Dropchecker {
            dropped: &mut dropped_b,
        });
        assert_eq!(dropped_a, false);
        assert_eq!(dropped_b, false);

        let mut itr = frame.downgrade().into_iter();
        drop(itr.next());
        assert_eq!(dropped_a, true);
        assert_eq!(dropped_b, false);
        drop(itr);
        assert_eq!(dropped_a, true);
        assert_eq!(dropped_b, true);
    }

    #[test]
    fn ring_frame_extend() {
        let v = vec![0, 1, 2, 3];

        let ring = FramedRing::<i32, Pow2Capacity>::new();
        let mut frame = ring.frame();
        frame.extend(v);

        for i in 0..4 {
            assert_eq!(*frame.as_ref().get(i).unwrap(), i as i32);
        }
    }
}
