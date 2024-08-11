// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_options.hpp"
#include "azure/storage/files/datalake/datalake_responses.hpp"
#include "azure/storage/files/datalake/datalake_service_client.hpp"

#include <azure/core/credentials/credentials.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/response.hpp>
#include <azure/storage/blobs/blob_container_client.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include <memory>
#include <string>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  class DataLakePathClient;
  class DataLakeFileClient;
  class DataLakeDirectoryClient;

  /** @brief The DataLakeFileSystemClient allows you to manipulate Azure Storage DataLake files.
   *
   */
  class DataLakeFileSystemClient final {
  public:
    /**
     * @brief Create from connection string
     * @param connectionString Azure Storage connection string.
     * @param fileSystemName The name of a file system.
     * @param options Optional parameters used to initialize the client.
     * @return FileSystemClient
     */
    static DataLakeFileSystemClient CreateFromConnectionString(
        const std::string& connectionString,
        const std::string& fileSystemName,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Shared key authentication client.
     * @param fileSystemUrl The URL of the file system this client's request targets.
     * @param credential The shared key credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileSystemClient(
        const std::string& fileSystemUrl,
        std::shared_ptr<StorageSharedKeyCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Bearer token authentication client.
     * @param fileSystemUrl The URL of the file system this client's request targets.
     * @param credential The token credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileSystemClient(
        const std::string& fileSystemUrl,
        std::shared_ptr<Core::Credentials::TokenCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Anonymous/SAS/customized pipeline auth.
     * @param fileSystemUrl The URL of the file system this client's request targets.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileSystemClient(
        const std::string& fileSystemUrl,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Create a DataLakeFileClient from current DataLakeFileSystemClient
     * @param fileName Name of the file within the file system.
     * @return DataLakeFileClient
     */
    DataLakeFileClient GetFileClient(const std::string& fileName) const;

    /**
     * @brief Create a DataLakeDirectoryClient from current DataLakeFileSystemClient
     * @param directoryName Name of the directory within the file system.
     * @return DataLakeDirectoryClient
     */
    DataLakeDirectoryClient GetDirectoryClient(const std::string& directoryName) const;

    /**
     * @brief Gets the filesystem's primary URL endpoint. This is the endpoint used for blob
     * storage available features in DataLake.
     *
     * @return The filesystem's primary URL endpoint.
     */
    std::string GetUrl() const { return m_blobContainerClient.GetUrl(); }

    /**
     * @brief Creates the file system.
     * @param options Optional parameters to create this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateFileSystemResult> containing the
     * information of create a file system.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::CreateFileSystemResult> Create(
        const CreateFileSystemOptions& options = CreateFileSystemOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Creates the file system if it does not exists.
     * @param options Optional parameters to create this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateFileSystemResult> containing the
     * information of create a file system. Only valid when successfully created the file system.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::CreateFileSystemResult> CreateIfNotExists(
        const CreateFileSystemOptions& options = CreateFileSystemOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the file system.
     * @param options Optional parameters to delete this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteFileSystemResult> containing the
     * information returned when deleting file systems.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::DeleteFileSystemResult> Delete(
        const DeleteFileSystemOptions& options = DeleteFileSystemOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the file system if it exists.
     * @param options Optional parameters to delete this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteFileSystemResult> containing the
     * information returned when deleting file systems. Only valid when successfully deleted the
     * file system.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::DeleteFileSystemResult> DeleteIfExists(
        const DeleteFileSystemOptions& options = DeleteFileSystemOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the metadata of file system.
     * @param metadata User-defined metadata to be stored with the filesystem. Note that the string
     *                 may only contain ASCII characters in the ISO-8859-1 character set.
     * @param options Optional parameters to set the metadata to this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::SetFileSystemMetadataResult> containing the
     * information returned when setting the metadata onto the file system.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::SetFileSystemMetadataResult> SetMetadata(
        Storage::Metadata metadata,
        const SetFileSystemMetadataOptions& options = SetFileSystemMetadataOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Gets the properties of file system.
     * @param options Optional parameters to get the metadata of this file system.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::FileSystemProperties> containing the
     * information when getting the file system's properties.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::FileSystemProperties> GetProperties(
        const GetFileSystemPropertiesOptions& options = GetFileSystemPropertiesOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Returns a sequence of paths in this file system. Enumerating the paths may make
     * multiple requests to the service while fetching all the values.
     * @param recursive If "true", all paths are listed; otherwise, only paths at the root of the
     *                  filesystem are listed.
     * @param options Optional parameters to list the paths in file system.
     * @param context Context for cancelling long running operations.
     * @return ListPathsPagedResponse describing the paths in this filesystem.
     * @remark This request is sent to dfs endpoint.
     */
    ListPathsPagedResponse ListPaths(
        bool recursive,
        const ListPathsOptions& options = ListPathsOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Gets the permissions for this file system. The permissions indicate whether
     * file system data may be accessed publicly.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A FileSystemAccessPolicy describing the container's access policy.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::FileSystemAccessPolicy> GetAccessPolicy(
        const GetFileSystemAccessPolicyOptions& options = GetFileSystemAccessPolicyOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the permissions for the specified file system. The permissions indicate
     * whether file system's data may be accessed publicly.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A SetFileSystemAccessPolicyResult describing the updated file system.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::SetFileSystemAccessPolicyResult> SetAccessPolicy(
        const SetFileSystemAccessPolicyOptions& options = SetFileSystemAccessPolicyOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Renames a file. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param fileName The file that gets renamed.
     * @param destinationFilePath The path of the file the source file is renaming to.
     * @param options Optional parameters to rename a file
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<DataLakeFileClient> The client targets the renamed file.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<DataLakeFileClient> RenameFile(
        const std::string& fileName,
        const std::string& destinationFilePath,
        const RenameFileOptions& options = RenameFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Renames a directory. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param directoryName The directory that gets renamed.
     * @param destinationDirectoryPath The destinationPath the source directory is renaming to.
     * @param options Optional parameters to rename a directory.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<DataLakeDirectoryClient> The client targets the renamed
     * directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<DataLakeDirectoryClient> RenameDirectory(
        const std::string& directoryName,
        const std::string& destinationDirectoryPath,
        const RenameDirectoryOptions& options = RenameDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Gets the paths that have recently been soft deleted in this file system.
     * @param options Optional parameters to list deleted paths.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<DataLakePathClient> The client targets the restored path.
     * @remark This request is sent to Blob endpoint.
     */
    ListDeletedPathsPagedResponse ListDeletedPaths(
        const ListDeletedPathsOptions& options = ListDeletedPathsOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Restores a soft deleted path.
     * @param deletedPath The path of the deleted path.
     * @param deletionId The deletion ID associated with the soft deleted path. You can get soft
     * deleted paths and their associated deletion IDs with ListDeletedPaths.
     * @param options Options to configure the restore operation.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<DataLakePathClient> The client targets the restored path.
     * @remark This request is sent to Blob endpoint.
     */
    Azure::Response<DataLakePathClient> UndeletePath(
        const std::string& deletedPath,
        const std::string& deletionId,
        const UndeletePathOptions& options = UndeletePathOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

  private:
    Azure::Core::Url m_fileSystemUrl;
    Blobs::BlobContainerClient m_blobContainerClient;
    std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> m_pipeline;
    _detail::DatalakeClientConfiguration m_clientConfiguration;

    explicit DataLakeFileSystemClient(
        Azure::Core::Url fileSystemUrl,
        Blobs::BlobContainerClient blobContainerClient,
        std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> pipeline,
        _detail::DatalakeClientConfiguration clientConfiguration)
        : m_fileSystemUrl(std::move(fileSystemUrl)),
          m_blobContainerClient(std::move(blobContainerClient)), m_pipeline(std::move(pipeline)),
          m_clientConfiguration(std::move(clientConfiguration))
    {
    }
    friend class DataLakeLeaseClient;
    friend class DataLakeServiceClient;
  };
}}}} // namespace Azure::Storage::Files::DataLake
