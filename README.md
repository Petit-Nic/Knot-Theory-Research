# Knot-Theory-Research
A repo containing the code developed in my knot theory research.

## Purpose
This code was created from scratch by me to assist in my mathematical research in knot theory. Due to my interest in finite-type invariants of virtual knots and tangles, the code mostly revolves around coding Gauss diagrams/codes for virtual knots and links, with optional labelings, and use said representations to compute the Affine Index Polynomial (as defined in the author's papers) for arbitrary examples and labelings.

## Contents
In its first commit, there is only one main file, GaussDiag.py, containing all the code for the above purposes. It is the author's intention to restructure this in a more appropriate python package, and potentially split the Gauss code generation and drawing from the invariant computations in due time.

## Future updates
Following my interests I would like to add the ability to process tangles to my code, and the associated AIP for tangles I recently defined. Tangles are typically represented by pictures instead of codes, so there are some challenges related to figuring out how to implement them, and if the tangle closes up to a knot I'd like to be able to reconstruct the Gauss code of the given knot.

I was also working on implementing the Cheng-Gao polynomial in my code, but I haven't touched that code in three years, so whether I'll get back to it is anybodys guess

## License
Copyright (C) 2020-2024 Nicolas Petit <petitnicola@gmail.com>

This file is part of the Gauss Diagram and AIP Invariant Python package project.

The Gauss Diagram and AIP Invariant Python package project cannot be copied and/or distributed without the express
permission of Nicolas Petit <petitnicola@gmail.com>.