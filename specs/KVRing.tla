---------------------------- MODULE KVRing ----------------------------
(* Model of the CACHEMODE-1 sliding-window KV ring protocol
   (kernels/attnf32.wgsl + kernels/headprep.wgsl + src/engine-a4b.js).

   Writer (headprep): position p's K/V land in slot p % RING. Positions are
   written strictly in order. Prefill runs in chunks of C tokens; ALL of a
   chunk's writes complete before ANY of its attention reads (separate GPU
   passes). Generation appends one position per step, read follows write.

   Reader (attnf32): for query position qp with newest written position mw,
   every slot t is scanned; its position is RECOVERED as
       ps(t) = mw - ((mw + RING - t) % RING)
   and the slot is dead iff ps > qp (causality) or ps + W <= qp (window).

   Checked invariants:
     SoundRec  (V-P1): the recovery formula returns exactly the newest
                position <= mw congruent to t mod RING — i.e. the slot's
                true content. A clobbered position can therefore only DROP
                OUT (recovered ps lands outside the window/causality and is
                marked dead); it can never be misread as stale K/V.
     Complete  (V-P2): every position in every query's live window
                {max(0,qp-W+1) .. qp} is recoverable from its slot at read
                time. This is what fails when RING is too small: a chunk's
                own late writes overwrite slots its early queries need.
                Paper bound: worst distance mw - (qp-W+1) = W+C-2, so
                Complete requires RING >= W+C-1. *)
EXTENDS Naturals

CONSTANTS W,        \* sliding window size
          C,        \* prefill chunk size (MPRE)
          RING,     \* ring slot count (production: W + 512)
          NCHUNKS,  \* prefill chunks to model
          GEN       \* generation steps after prefill

ASSUME W >= 1 /\ C >= 1 /\ RING >= 1 /\ NCHUNKS >= 1 /\ GEN >= 0

TOTAL == NCHUNKS * C

VARIABLES mw,     \* newest position written into the ring
          base,   \* base position of the chunk currently being read
          phase   \* "pre" (chunk reads) / "gen" (decode reads) / "done"

vars == <<mw, base, phase>>

RecPs(t, m) == m - ((m + RING - t) % RING)

WinLo(qp) == IF qp + 1 >= W THEN qp + 1 - W ELSE 0

\* every needed position of query qp is recoverable when mw = m
Served(qp, m) == \A p \in WinLo(qp)..qp : RecPs(p % RING, m) = p

Init == mw = C - 1 /\ base = 0 /\ phase = "pre"

NextChunk == /\ phase = "pre"
             /\ IF base + C < TOTAL
                THEN /\ base' = base + C
                     /\ mw'   = base + 2 * C - 1
                     /\ phase' = "pre"
                ELSE IF GEN > 0
                THEN /\ mw' = mw + 1 /\ base' = base /\ phase' = "gen"
                ELSE /\ phase' = "done" /\ UNCHANGED <<mw, base>>

GenStep == /\ phase = "gen"
           /\ IF mw < TOTAL + GEN - 1
              THEN mw' = mw + 1 /\ base' = base /\ phase' = "gen"
              ELSE phase' = "done" /\ UNCHANGED <<mw, base>>

Next == NextChunk \/ GenStep

Spec == Init /\ [][Next]_vars

\* --- V-P1: recovery soundness -------------------------------------------
SoundRec ==
  \A t \in 0..(RING - 1) :
    t <= mw =>
      LET S == {p \in 0..mw : p % RING = t}
          r == RecPs(t, mw)
      IN r \in S /\ \A p \in S : p <= r

\* --- V-P2: window completeness -------------------------------------------
Complete ==
  IF phase = "pre"
  THEN \A qp \in base..(base + C - 1) : Served(qp, mw)
  ELSE IF phase = "gen" THEN Served(mw, mw)
  ELSE TRUE
========================================================================
