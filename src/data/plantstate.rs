use std::{error::Error, path::Path, fs::File, io::{self, Read}};

use serde::{Serialize, Deserialize};
use rand::{Rng, thread_rng};
use unda::core::data::input::Input;

#[derive(Default, Serialize, Deserialize)]
pub struct PlantState {
    id: u64,
    sun_exp: f32,
    hrs_since_last_water: f32,
    soil_moisture: f32,
    water: Vec<f32>,
    occurences: u16
}

impl PlantState {
    pub fn new(id: u64) -> Self {
        let mut rng = thread_rng();

        PlantState { 
            id,
            sun_exp: rng.gen_range(0.0..1.0),
            hrs_since_last_water: rng.gen_range(0.0..10.0),
            soil_moisture: rng.gen_range(0.0..1.0),
            water: vec![],
            occurences: 1
        }
    }

    pub fn save_plants<P: AsRef<Path>>(data: &[PlantState], path: P) -> Result<(), Box<dyn Error>> {
        let file = File::create(path)?;
        serde_json::to_writer(file, &data)?;
        Ok(())
    }
    pub fn get_plants<P: AsRef<Path>>(path: P) -> Result<Vec<PlantState>, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut buf_reader = io::BufReader::new(file);
        let mut contents = String::new();
        buf_reader.read_to_string(&mut contents)?;

        Ok(serde_json::from_str(&contents)?)
    }
    pub fn output(&self) -> f32 {
        if self.water.len() == 0 {
            return 0f32
        }
        self.water.iter().sum::<f32>() / self.water.len() as f32
    }
    pub fn data_tuple(&self) -> (Vec<f32>, Vec<f32>) {
        (self.to_param(), vec![self.output()])
    }
}

impl Input for PlantState {
    fn to_param(&self) -> Vec<f32> {
        vec![self.sun_exp,
            self.soil_moisture,
            self.hrs_since_last_water
        ]
    }
    fn to_param_2d(&self) -> Vec<Vec<f32>> {
        vec![self.to_param()]
    }

    fn to_param_3d(&self) -> Vec<Vec<Vec<f32>>> {
        vec![vec![self.to_param()]]
    }
    
    fn shape(&self) -> (usize, usize, usize) {
        (3,1,1)
    }
    fn to_box(&self) -> Box<dyn Input> {
        Box::new(self.to_param())        
    }
}
