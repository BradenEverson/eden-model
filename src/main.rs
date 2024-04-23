use std::{error::Error, path::Path};

use eden_ai::data::plantstate::{PlantState, SunState};
use unda::core::{data::input::Input, network::Network, layer::{layers::{InputTypes, LayerTypes}, methods::{activations::Activations, errors::ErrorTypes}}};

fn main() -> Result<(), Box<dyn Error>>{
    loop {
        let sun = get_sun("sun.json")?; 
        let mut inputs: Vec<&dyn Input> = vec![];
        let mut outputs = vec![];

        let data_tuple: Vec<(Vec<f32>, Vec<f32>)> = sun.iter().map(|sun| sun.data_tuple()).collect();

        data_tuple.iter().for_each(|(inp, out)| {
            inputs.push(inp);
            outputs.push(out.clone());
        });

        let mut sun_model = Network::new(180);

        sun_model.set_input(InputTypes::DENSE(data_tuple[0].0.len()));

        sun_model.add_layer(LayerTypes::DENSE(64, Activations::SIGMOID, 0.001));
        sun_model.add_layer(LayerTypes::DENSE(16, Activations::SIGMOID, 0.001));
        sun_model.add_layer(LayerTypes::DENSE(8, Activations::SIGMOID, 0.001));

        sun_model.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

        sun_model.compile();

        sun_model.fit(&inputs, &outputs, 4,
            ErrorTypes::MeanAbsolute);

        sun_model.serialize_unda_fmt(&format!("sun_data_{:.2}_relu.unda", sun_model.loss * 100f32));

        /*
        let plants = get_data("plants.json")?;

        let mut inputs: Vec<&dyn Input> = vec![];
        let mut outputs = vec![];

        let data_tuple: Vec<(Vec<f32>, Vec<f32>)> = 
            plants.iter().map(|plant| plant.data_tuple()).collect();

        data_tuple.iter().for_each(|(inp, out)| {
            inputs.push(inp);
            outputs.push(out.clone());
        });

        let mut plant_model = Network::new(128);

        plant_model.set_input(InputTypes::DENSE(data_tuple[0].0.len()));

        plant_model.add_layer(LayerTypes::DENSE(512, Activations::RELU, 0.001));
        plant_model.add_layer(LayerTypes::DENSE(128, Activations::RELU, 0.001));
        plant_model.add_layer(LayerTypes::DENSE(32, Activations::RELU, 0.001));

        plant_model.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

        plant_model.compile();

        plant_model.fit(&inputs, &outputs, 1000, 
            ErrorTypes::MeanAbsolute);

        plant_model.serialize_unda_fmt(&format!("plant_data_{:.2}_sigmoid.unda", plant_model.loss * 100f32));
        */
    }
}

fn create_data_sun() -> Result<(), Box<dyn Error>>{
    let mut plants: Vec<SunState> = vec![];
    for i in 0..200 {
        plants.push(SunState::new(i));
    }
    SunState::save(&plants, "sundata.json")?;
    println!("Done");
    Ok(())
}

fn create_data() -> Result<(), Box<dyn Error>>{
    let mut plants: Vec<PlantState> = vec![];
    for i in 0..200 {
        plants.push(PlantState::new(i));
    }
    PlantState::save_plants(&plants, "plantdata.json")?;
    println!("Done");
    Ok(())
}

fn get_data<P: AsRef<Path>>(path: P) -> Result<Vec<PlantState>, Box<dyn Error>> {
    PlantState::get_plants(path)
}


fn get_sun<P: AsRef<Path>>(path: P) -> Result<Vec<SunState>, Box<dyn Error>> {
    SunState::get(path)
}
