use std::{error::Error, path::Path};

use eden_ai::data::plantstate::PlantState;

fn main() {

}

fn create_data() -> Result<(), Box<dyn Error>>{
    let mut plants: Vec<PlantState> = vec![];
    for i in 0..100 {
        plants.push(PlantState::new(i));
    }
    PlantState::save_plants(&plants, "plantdata.json")?;
    println!("Done");
    Ok(())
}

fn get_data<P: AsRef<Path>>(path: P) -> Result<Vec<PlantState>, Box<dyn Error>> {
    PlantState::get_plants(path)
}
