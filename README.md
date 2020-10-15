# FrameRing

FrameRing is a queue-of-queues designed for memory efficiency & locality.
FrameRing can be used to create looping software which avoids memory
allocations for temporary scratch vectors and supports memory locality even
across multiple different vectors.

## Overview

Frames created in the same FrameRing share the same backing memory, allowing
for the creation of large amounts of temporary lists without needing a memory
allocation each time. Many FrameRing operations are constant time, including
creating and dropping frames, appending to the mutable frame, iterating over
any frame, and random access into any frame.

FrameRing is able to reuse memory previously freed from dropping older frames
for elements appended to new frames. Only the front frame (the mutable frame)
is able to be appended to at any time, but elements from previous frames are
able to be accessed until the frame itself is dropped, making the structure
ideal for storing lists for use in computations which mimic map reduce
operations.

## Example

```
let mut ring = FramedRing::<i32, Pow2Capacity>::new();
let mut frame = ring.frame();
frame.push(1);
let (frame_ro, mut frame2) = frame.next();
// frame_ro now contains [1], but frame_ro is incapable of additional pushes
// However, frame2 is now able to be appended to.
frame2.push(2);
// frame_ro can be dropped whenever without requiring frame2 to be dropped
drop(frame_ro);
// ... while still maintaining the elements in frame2
frame2.as_ref().get(0) // Returns Some(2)
```

## License

GPL v3 - see header at top of [src/lib.rs](./src/lib.rs).

## Contributing

Feel free to submit any changes as a PR request, as long as the contribution is
licensed under GPL v3 without any additional terms.
