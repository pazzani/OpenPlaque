# Left proximal-trunk local multi-direction reacquisition v1

## Motivation

The expanded-field endpoint experiment ruled out the narrow source-field crop as the main failure mode. It used an 18-mm field margin, had no outside-field rejections, and still produced no valid extension. Dense QC showed that its chosen path remained source-supported for only about 3 mm before lumen geometry deteriorated.

The first orthogonal plane of that run also failed despite using the same physical endpoint that had passed in the preceding dense-QC continuation experiment. This is consistent with immediate path/tangent diversion into an adjacent or crossing structure rather than loss of source signal at the validated endpoint itself.

## Prospective question

Can a short, local, source-led search recover a valid continuation if we do not force every discovery step to monotonically approach the aorta and do not commit to a single terminal direction?

## Design

- Fresh branch from frozen baseline `0593b453959f5a353d644267fbeef24b514ef4d7`.
- Prerequisites are the positive 20-mm proximal-trunk continuation result and the completed expanded-field negative result.
- Four seeds are placed at 0.0, 0.8, 1.6 and 2.4 mm proximal to the validated endpoint.
- Each seed generates multiple forward source-supported initial directions. The top five spatially distinct directions are retained.
- Each direction runs a 10-mm local beam search.
- During local discovery, aortic distance is **not** used as an acceptance gate or ranking term.
- HU, vesselness, coronary-mask proximity, turning and loop gates remain active.
- Aortic distance is measured only after candidate discovery.
- Candidate paths undergo the same dense 0.4-mm orthogonal source-CCTA plane QC used for the preceding continuation experiment.
- A local continuation requires at least 5 mm retained length and at least 80% plane-pass fraction, with truncation at three consecutive failed planes.
- The canonical RCA again serves as a positive control for plane QC.

## Interpretation boundary

A positive result means only that a source-supported local vessel continuation was recovered beyond the validated proximal-trunk endpoint. It does not establish a left-main identity, an aortic bridge, LCX topology, or modify the frozen master anatomy.
