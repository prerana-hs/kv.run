use super::ShardedClient;
use futures::future::join_all;
use tracing::instrument;
use crate::pb::generate::v2::AdapterControlResponse;

impl ShardedClient {
    /// Get model health
    #[instrument(skip(self))]
    pub async fn adapter_control(
        &mut self,
        lora_ids: Option<String>,
        operation: String
    ) -> crate::Result<AdapterControlResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.adapter_control(lora_ids.clone(), operation.clone())))
            .collect();
        join_all(futures).await.pop().unwrap()
    }
}