use super::Client;
use crate::pb::generate::v2::*;
use crate::Result;
use grpc_metadata::InjectTelemetryContext;
use tracing::instrument;

impl Client {
    /// Get model health
    #[instrument(skip(self))]
    pub async fn adapter_control(
        &mut self,
        lora_ids: Option<String>,
        operation: String
    ) -> Result<AdapterControlResponse> {
        let request = tonic::Request::new(AdapterControlRequest { lora_ids, operation }).inject_context();
        let response = self.stub.adapter_control(request).await?.into_inner();
        Ok(response)
    }
}