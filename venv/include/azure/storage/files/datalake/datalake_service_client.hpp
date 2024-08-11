// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_options.hpp"
#include "azure/storage/files/datalake/datalake_responses.hpp"

#include <azure/core/credentials/credentials.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/response.hpp>
#include <azure/storage/blobs/blob_service_client.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include <memory>
#include <string>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  class DataLakeFileSystemClient;

  /** @brief The DataLakeServiceClient allows you to manipulate Azure Storage DataLake files.
   *
   */
  class DataLakeServiceClient final {
  public:
    /**
     * @brief Create from connection string
     * @param connectionString Azure Storage connection string.
     * @param options Optional parameters used to initialize the client.
     * @return DataLakeServiceClient
     */
    static DataLakeServiceClient CreateFromConnectionString(
        const std::string& connectionString,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Shared key authentication client.
     * @param serviceUrl The service URL this client's request targets.
     * @param credential The shared key credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeServiceClient(
        const std::string& serviceUrl,
        std::shared_ptr<StorageSharedKeyCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Bearer token authentication client.
     * @param serviceUrl The service URL this client's request targets.
     * @param credential The token credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeServiceClient(
        const std::string& serviceUrl,
        std::shared_ptr<Core::Credentials::TokenCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Anonymous/SAS/customized pipeline auth.
     * @param serviceUrl The service URL this client's request targets.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeServiceClient(
        const std::string& serviceUrl,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Create a FileSystemClient from current DataLakeServiceClient
     * @param fileSystemName The name of the file system.
     * @return FileSystemClient
     */
    DataLakeFileSystemClient GetFileSystemClient(const std::string& fileSystemName) const;

    /**
     * @brief Gets the datalake service's primary URL endpoint. This is the endpoint used for blob
     * storage available features in DataLake.
     *
     * @return The datalake service's primary URL endpoint.
     */
    std::string GetUrl() const { return m_blobServiceClient.GetUrl(); }

    /**
     * @brief Returns a sequence of file systems in the storage account. Enumerating the file
     * systems may make multiple requests to the service while fetching all the values. File systems
     * are ordered lexicographically by name.
     * @param options Optional parameters to list the file systems.
     * @param context Context for cancelling long running operations.
     * @return ListFileSystemsPagedResponse describing the file systems in the storage account.
     * @remark This request is sent to blob endpoint.
     */
    ListFileSystemsPagedResponse ListFileSystems(
        const ListFileSystemsOptions& options = ListFileSystemsOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Retrieves a key that can be used to delegate Active Directory authorization to
     * shared access signatures.
     *
     * @param expiresOn Expiration of the key's validity. The time should be specified in UTC, and
     * will be truncated to second.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::UserDelegationKey> containing the user delegation key.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::UserDelegationKey> GetUserDelegationKey(
        const Azure::DateTime& expiresOn,
        const GetUserDelegationKeyOptions& options = GetUserDelegationKeyOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return m_blobServiceClient.GetUserDelegationKey(expiresOn, options, context);
    }

    /**
     * @brief Sets properties for a storage account's Blob service endpoint, including
     * properties for Storage Analytics, CORS (Cross-Origin Resource Sharing) rules and soft delete
     * settings. You can also use this operation to set the default request version for all incoming
     * requests to the DataLake service that do not have a version specified.
     *
     * @param properties The DataLake service properties.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A SetServicePropertiesResult on successfully setting the properties.
     */
    Azure::Response<Models::SetServicePropertiesResult> SetProperties(
        Models::DataLakeServiceProperties properties,
        const SetServicePropertiesOptions& options = SetServicePropertiesOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return m_blobServiceClient.SetProperties(properties, options, context);
    }

    /**
     * @brief Gets the properties of a storage account's datalake service, including properties
     * for Storage Analytics and CORS (Cross-Origin Resource Sharing) rules.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A DataLakeServiceProperties describing the service properties.
     */
    Azure::Response<Models::DataLakeServiceProperties> GetProperties(
        const GetServicePropertiesOptions& options = GetServicePropertiesOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return m_blobServiceClient.GetProperties(options, context);
    }

  private:
    Azure::Core::Url m_serviceUrl;
    Blobs::BlobServiceClient m_blobServiceClient;
    std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> m_pipeline;
    _detail::DatalakeClientConfiguration m_clientConfiguration;
  };
}}}} // namespace Azure::Storage::Files::DataLake
