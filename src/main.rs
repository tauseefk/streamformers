pub mod prelude {
    pub use std::env;
    pub use std::pin::Pin;
    pub use std::sync::{mpsc::SyncSender, Arc, Mutex, OnceLock};

    pub use bytes::{BufMut, BytesMut};
    pub use futures::{pin_mut, Future, StreamExt};
    pub use llm::{models::Llama, Model, Prompt};
    pub use rand;
    pub use std::{sync::mpsc::sync_channel, thread};
    pub use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        sync::mpsc::{channel, Receiver},
    };
}

use crate::prelude::*;

fn init_llm() -> Result<Llama, &'static str> {
    let current_dir = env::current_dir().unwrap();
    let model_path = format!(
        "{}/models/open_llama_3b-q4_0-ggjt.bin",
        &current_dir.display()
    );

    // load a GGML model from disk
    let llama = llm::load::<llm::models::Llama>(
        // path to GGML file
        std::path::Path::new(model_path.as_str()),
        // Tokenizer
        llm::TokenizerSource::Embedded,
        // llm::ModelParameters
        llm::ModelParameters {
            use_gpu: true,
            ..Default::default()
        },
        // load progress callback
        llm::load_progress_callback_stdout,
    );

    match llama {
        Ok(model) => Ok(model),
        Err(err) => {
            println!("{err}");
            Err("Failed to load model")
        }
    }
}

fn infer(
    rx_infer: std::sync::mpsc::Receiver<String>,
    tx_callback: tokio::sync::mpsc::Sender<llm::InferenceResponse>,
) -> Result<(), &'static str> {
    let llm_model = init_llm().unwrap();

    while let Ok(msg) = rx_infer.recv() {
        let mut session = llm_model.start_session(Default::default());

        let _res = session.infer::<std::convert::Infallible>(
            &llm_model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: Prompt::Text(&msg),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: Some(1000),
            },
            &mut Default::default(),
            |r| match r {
                llm::InferenceResponse::EotToken => {
                    let _ = tx_callback.blocking_send(r);
                    Ok(llm::InferenceFeedback::Halt)
                }
                _ => {
                    let _ = tx_callback.blocking_send(r);
                    Ok(llm::InferenceFeedback::Continue)
                }
            },
        );

        let _ = tx_callback.blocking_send(llm::InferenceResponse::EotToken);
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    let (tx_infer, rx_infer) = sync_channel::<String>(3);
    let (tx_callback, mut rx_callback) = channel::<llm::InferenceResponse>(3);

    thread::spawn(move || {
        let _ = infer(rx_infer, tx_callback);
    });

    let prompt = "I'm a potato".to_string();

    let _ = tx_infer.send(prompt.to_string());

    let inference_stream = async_stream::stream! {
        let mut buf = BytesMut::with_capacity(1024);

        while let Some(msg) = &rx_callback.recv().await {
            match msg {
                llm::InferenceResponse::PromptToken(val)
                | llm::InferenceResponse::InferredToken(val) => {
                    buf.put(val.as_bytes());
                    yield buf.clone();
                }
                _ => break,
            }
        }
    };

    pin_mut!(inference_stream);

    while let Some(value) = inference_stream.next().await {
        println!("{:?}", value);
    }
}
