# S11 FFT legend matcher

Uses S02 PELT intervals plus isolated dark horizontal-rule proposals. Interior
swatches and interval windows use Hann-windowed FFT magnitudes pooled by angle
and frequency relative to the dominant period; adjacent radial-bin shifts
provide coarse scale jitter. Ink fraction and Lab histogram/median contribute
to the distance, with colour suppressed for monochrome columns. Median window
distances select a swatch, with an Unknown option for very small top-two
margins or poor absolute fit. Unknowns are grouped by feature distance.
Pipeline images show swatch and interval spectra, interval margins, the
distance matrix, and final labels. The final synthetic run averaged 0.738
score, 0.864 row accuracy, and 0.847 boundary F1 over 39 windows.

The synthetic generator repeats the legend pixels exactly, so it does not
measure realistic legend-to-column changes in scale, ink, or rendering. The
first real log06 review showed high boundary agreement and clustering, but
systematically wrong lithology names; distinct patterns can collide when the
legend hatch scale or appearance changes. The legend parser can also return
labels that differ from the gold spelling. Colour and blank handling need
additional validation, as do tiny beds, text overlays, and narrow strips.
The final gold log06 run scored 0.110, with 0.129 row accuracy, 0.848
boundary F1, and 0.830 cluster ARI; the other evaluation windows lack gold.

Potential follow-ups: realistic held-out synthetic phase/scale augmentations,
multiscale local descriptors and swatch-to-column registration, calibrated
rejection on independent examples, and better separation of hatch strokes from
true horizontal boundary rules.
