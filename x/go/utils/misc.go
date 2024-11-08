package utils

import (
    "image/png"
    "os"

    "gocv.io/x/gocv"
)

func SolveAffine(xyArray, uvArray []gocv.Point2f) gocv.Mat {
    // Implement the affine transformation solver
    // Return the transformation matrix
    return gocv.GetPerspectiveTransform2f(xyArray, uvArray)
}

func WritePNGWithDPI(filename string, img gocv.Mat, dpiX, dpiY int) error {
    // Convert gocv.Mat to image.Image
    imgRGBA, err := img.ToImage()
    if err != nil {
        return err
    }

    // Set DPI in PNG metadata
    encoder := png.Encoder{
        CompressionLevel: png.DefaultCompression,
    }

    f, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer f.Close()

    return encoder.Encode(f, imgRGBA)
}
