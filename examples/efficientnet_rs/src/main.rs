mod net;

fn main() {
    let file: String = std::env::args().nth(1).unwrap();

    let image = image::load_from_memory_with_format(
        &std::fs::read(&file).unwrap(),
        image::ImageFormat::Jpeg,
    )
    .unwrap();

    let image = image.resize_exact(255, 255, image::imageops::FilterType::Nearest);
    let image = image.to_rgb32f();

    unsafe {
        // resize to input[1,3,224,224] and rescale
        for y in 0..224usize {
            for x in 0..224usize {
                let px = image.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    net::input[c * 224 * 224 + y * 224 + x] = px.0[c];
                }
            }
        }

        net::net();

        let best = net::lbls
            .iter()
            .zip(net::outputs.iter())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        println!("image is: {}", best.0);
    }
}
