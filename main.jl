begin
    using Images

    image = load("./assets/download.jpeg")
    height, width = size(image)

    println(typeof(image))

    
end