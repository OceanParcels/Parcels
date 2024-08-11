// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_options.hpp"
#include "azure/storage/files/datalake/datalake_path_client.hpp"
#include "azure/storage/files/datalake/datalake_responses.hpp"

#include <azure/core/credentials/credentials.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/response.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  /** @brief DataLake Directory Client.
   */
  class DataLakeDirectoryClient final : public DataLakePathClient {
  public:
    /**
     * @brief Create from connection string
     * @param connectionString Azure Storage connection string.
     * @param fileSystemName The name of a file system.
     * @param directoryName The name of a directory within the file system.
     * @param options Optional parameters used to initialize the client.
     * @return DataLakeDirectoryClient
     */
    static DataLakeDirectoryClient CreateFromConnectionString(
        const std::string& connectionString,
        const std::string& fileSystemName,
        const std::string& directoryName,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Shared key authentication client.
     * @param directoryUrl The URL of the file system this client's request targets.
     * @param credential The shared key credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeDirectoryClient(
        const std::string& directoryUrl,
        std::shared_ptr<StorageSharedKeyCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Bearer token authentication client.
     * @param directoryUrl The URL of the file system this client's request targets.
     * @param credential The token credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeDirectoryClient(
        const std::string& directoryUrl,
        std::shared_ptr<Core::Credentials::TokenCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Anonymous/SAS/customized pipeline auth.
     * @param directoryUrl The URL of the file system this client's request targets.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeDirectoryClient(
        const std::string& directoryUrl,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Create a FileClient from current DataLakeDirectoryClient
     * @param fileName Name of the file under the directory.
     * @return FileClient
     */
    DataLakeFileClient GetFileClient(const std::string& fileName) const;

    /**
     * @brief Create a DataLakeDirectoryClient from current DataLakeDirectoryClient
     * @param subdirectoryName Name of the directory under the current directory.
     * @return DataLakeDirectoryClient
     */
    DataLakeDirectoryClient GetSubdirectoryClient(const std::string& subdirectoryName) const;

    /**
     * @brief Gets the directory's primary URL endpoint. This is the endpoint used for blob
     * storage available features in DataLake.
     *
     * @return The directory's primary URL endpoint.
     */
    std::string GetUrl() const { return m_blobClient.GetUrl(); }

    /**
     * @brief Create a directory. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param options Optional parameters to create the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateDirectoryResult> containing the
     * information of the created directory
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreateDirectoryResult> Create(
        const CreateDirectoryOptions& options = CreateDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return DataLakePathClient::Create(Models::PathResourceType::Directory, options, context);
    }

    /**
     * @brief Create a directory. If it already exists, nothing will happen.
     * @param options Optional parameters to create the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateDirectoryResult> containing the
     * information of the created directory
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreateDirectoryResult> CreateIfNotExists(
        const CreateDirectoryOptions& options = CreateDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return DataLakePathClient::CreateIfNotExists(
          Models::PathResourceType::Directory, options, context);
    }

    /**
     * @brief Renames a file. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param fileName The file that gets renamed.
     * @param destinationFilePath The path of the file the source file is renaming to. The
     * destination is an absolute path under the root of the file system, without leading slash.
     * @param options Optional parameters to rename a file.
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
     * @param subdirectoryName The subdirectory that gets renamed.
     * @param destinationDirectoryPath The destinationPath the source subdirectory is renaming to.
     * The destination is an absolute path under the root of the file system, without leading slash.
     * @param options Optional parameters to rename a directory.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<DataLakeDirectoryClient> The client targets the renamed
     * directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<DataLakeDirectoryClient> RenameSubdirectory(
        const std::string& subdirectoryName,
        const std::string& destinationDirectoryPath,
        const RenameSubdirectoryOptions& options = RenameSubdirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the empty directory. Throws exception if directory is not empty.
     * @param options Optional parameters to delete the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteDirectoryResult> containing the information
     * returned when deleting the directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteDirectoryResult> DeleteEmpty(
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return this->Delete(false, options, context);
    }

    /**
     * @brief Deletes the empty directory if it already exists. Throws exception if directory is not
     * empty.
     * @param options Optional parameters to delete the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteDirectoryResult> containing the information
     * returned when deleting the directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteDirectoryResult> DeleteEmptyIfExists(
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return this->DeleteIfExists(false, options, context);
    }

    /**
     * @brief Deletes the directory and all its subdirectories and files.
     * @param options Optional parameters to delete the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteDirectoryResult> containing the information
     * returned when deleting the directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteDirectoryResult> DeleteRecursive(
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return this->Delete(true, options, context);
    }

    /**
     * @brief Deletes the directory and all its subdirectories and files if the directory exists.
     * @param options Optional parameters to delete the directory the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteDirectoryResult> containing the information
     * returned when deleting the directory.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteDirectoryResult> DeleteRecursiveIfExists(
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return this->DeleteIfExists(true, options, context);
    }

    /**
     * @brief Returns a sequence of paths in this directory. Enumerating the paths may make multiple
     * requests to the service while fetching all the values.
     * @param recursive If "true", all paths are listed; otherwise, the list will only
     *                  include paths that share the same root.
     * @param options Optional parameters to list the paths in file system.
     * @param context Context for cancelling long running operations.
     * @return ListPathsPagedResponse describing the paths in a directory.
     * @remark This request is sent to dfs endpoint.
     */
    ListPathsPagedResponse ListPaths(
        bool recursive,
        const ListPathsOptions& options = ListPathsOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

  private:
    explicit DataLakeDirectoryClient(
        Azure::Core::Url directoryUrl,
        Blobs::BlobClient blobClient,
        std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> pipeline,
        _detail::DatalakeClientConfiguration clientConfiguration)
        : DataLakePathClient(
            std::move(directoryUrl),
            std::move(blobClient),
            pipeline,
            std::move(clientConfiguration))
    {
    }

    Azure::Response<Models::DeleteDirectoryResult> Delete(
        bool recursive,
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    Azure::Response<Models::DeleteDirectoryResult> DeleteIfExists(
        bool recursive,
        const DeleteDirectoryOptions& options = DeleteDirectoryOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    friend class DataLakeFileSystemClient;
  };
}}}} // namespace Azure::Storage::Files::DataLake
