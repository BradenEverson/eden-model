use std::{error::Error, path::Path};

use eden_ai::data::plantstate::PlantState;
use unda::core::{data::input::Input, network::Network, layer::{layers::{InputTypes, LayerTypes}, methods::activations::Activations}};

fn main() -> Result<(), Box<dyn Error>>{
    let plants = get_data("filled_plants.json")?;
    let mut inputs: Vec<&dyn Input> = vec![];
    let mut outputs = vec![];

    let data_tuple: Vec<(Vec<f32>, Vec<f32>)> = 
        plants.iter().map(|plant| plant.data_tuple()).collect();

    data_tuple.iter().for_each(|(inp, out)| {
        inputs.push(inp);
        outputs.push(out);
    });

    let mut plant_model = Network::new(128);
    
    plant_model.set_input(InputTypes::DENSE(data_tuple[0].0.len()));

    plant_model.add_layer(LayerTypes::DENSE(16, Activations::RELU, 0.001));
    plant_model.add_layer(LayerTypes::DENSE(32, Activations::RELU, 0.001));

    plant_model.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));

    plant_model.compile();

    plant_model.serialize_unda_fmt("plant_model_3_16_32_1.unda");

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
