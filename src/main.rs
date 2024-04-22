use std::{error::Error, path::Path};

use eden_ai::data::plantstate::PlantState;
use unda::core::{data::input::Input, network::Network, layer::{layers::{InputTypes, LayerTypes}, methods::{activations::Activations, errors::ErrorTypes}}};

fn main() -> Result<(), Box<dyn Error>>{
    loop {
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

        plant_model.add_layer(LayerTypes::DENSE(64, Activations::SIGMOID, 0.001));
        plant_model.add_layer(LayerTypes::DENSE(128, Activations::SIGMOID, 0.001));
        plant_model.add_layer(LayerTypes::DENSE(32, Activations::SIGMOID, 0.001));

        plant_model.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

        plant_model.compile();

        plant_model.fit(&inputs, &outputs, 5, ErrorTypes::CategoricalCrossEntropy);

        plant_model.serialize_unda_fmt(&format!("plant_data_{:.2}_sigmoid.unda", plant_model.loss * 100f32));
    }
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
