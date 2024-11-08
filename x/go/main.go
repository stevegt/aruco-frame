package main

import (
    "syscall/js"
)

func main() {
    c := make(chan struct{}, 0)
    js.Global().Set("ProcessImage", js.FuncOf(ProcessImage))
    <-c
}

func ProcessImage(this js.Value, args []js.Value) interface{} {
    // Get image data from JavaScript
    data := args[0]
    width := args[1].Int()
    height := args[2].Int()
    channels := args[3].Int()

    // Convert JS TypedArray to Go byte slice
    imgData := make([]byte, data.Get("byteLength").Int())
    js.CopyBytesToGo(imgData, data)

    // Process image using OpenCV
    processedData := processImage(imgData, width, height, channels)

    // Convert Go byte slice back to JS Uint8Array
    uint8Array := js.Global().Get("Uint8Array").New(len(processedData))
    js.CopyBytesToJS(uint8Array, processedData)
    return uint8Array
}
