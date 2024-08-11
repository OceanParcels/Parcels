// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_options.hpp"
#include "azure/storage/files/datalake/datalake_path_client.hpp"
#include "azure/storage/files/datalake/datalake_responses.hpp"

#include <azure/core/credentials/credentials.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/response.hpp>
#include <azure/storage/blobs/block_blob_client.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  /** @brief The DataLakeFileClient allows you to manipulate Azure Storage DataLake files.
   *
   */
  class DataLakeFileClient final : public DataLakePathClient {
  public:
    /**
     * @brief Create from connection string
     * @param connectionString Azure Storage connection string.
     * @param fileSystemName The name of a file system.
     * @param fileName The name of a file within the file system.
     * @param options Optional parameters used to initialize the client.
     * @return DataLakeFileClient
     */
    static DataLakeFileClient CreateFromConnectionString(
        const std::string& connectionString,
        const std::string& fileSystemName,
        const std::string& fileName,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Shared key authentication client.
     * @param fileUrl The URL of the file this client's request targets.
     * @param credential The shared key credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileClient(
        const std::string& fileUrl,
        std::shared_ptr<StorageSharedKeyCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Bearer token authentication client.
     * @param fileUrl The URL of the file this client's request targets.
     * @param credential The token credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileClient(
        const std::string& fileUrl,
        std::shared_ptr<Core::Credentials::TokenCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Anonymous/SAS/customized pipeline auth.
     * @param fileUrl The URL of the file this client's request targets.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakeFileClient(
        const std::string& fileUrl,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Gets the file's primary URL endpoint. This is the endpoint used for blob
     * storage available features in DataLake.
     *
     * @return The file's primary URL endpoint.
     */
    std::string GetUrl() const { return m_blobClient.GetUrl(); }

    /**
     * @brief Uploads data to be appended to a file. Data can only be appended to a file.
     * @param content The data to be appended.
     * @param offset This parameter allows the caller to upload data in parallel and control
     *                 the order in which it is appended to the file.
     *                 The value must be the offset where the data is to be appended.
     *                 Uploaded data is not immediately flushed, or written, to the file. To flush,
     *                 the previously uploaded data must be contiguous, the offset parameter must
     *                 be specified and equal to the length of the file after all data has been
     *                 written, and there must not be a request entity body included with the
     *                 request.
     * @param options Optional parameters to append data to the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::AppendFileResult> containing the
     * information returned when appending some data to the path.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::AppendFileResult> Append(
        Azure::Core::IO::BodyStream& content,
        int64_t offset,
        const AppendFileOptions& options = AppendFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Flushes previous uploaded data to a file.
     * @param position This parameter allows the caller to upload data in parallel and control
     *                 the order in which it is appended to the file.
     *                 The value must be the offset where the data is to be appended.
     *                 Uploaded data is not immediately flushed, or written, to the file. To flush,
     *                 the previously uploaded data must be contiguous, the offset parameter must
     *                 be specified and equal to the length of the file after all data has been
     *                 written, and there must not be a request entity body included with the
     *                 request.
     * @param options Optional parameters to flush data to the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::FlushFileResult> containing the information
     * returned when flushing the data appended to the path.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::FlushFileResult> Flush(
        int64_t position,
        const FlushFileOptions& options = FlushFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Create a file. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param options Optional parameters to create the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateFileResult> containing the information
     * returned when creating the file.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreateFileResult> Create(
        const CreateFileOptions& options = CreateFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return DataLakePathClient::Create(Models::PathResourceType::File, options, context);
    }

    /**
     * @brief Create a file. If it already exists, it will remain unchanged.
     * @param options Optional parameters to create the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreateFileResult> containing the information
     * returned when creating the file.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreateFileResult> CreateIfNotExists(
        const CreateFileOptions& options = CreateFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return DataLakePathClient::CreateIfNotExists(
          Models::PathResourceType::File, options, context);
    }

    /**
     * @brief Deletes the file.
     * @param options Optional parameters to delete the file the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteFileResult>
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteFileResult> Delete(
        const DeleteFileOptions& options = DeleteFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the file if it already exists.
     * @param options Optional parameters to delete the file the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeleteFileResult>
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeleteFileResult> DeleteIfExists(
        const DeleteFileOptions& options = DeleteFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Download the contents of a file. For download operations, range requests are
     * supported.
     * @param options Optional parameters to download the content from the resource the path points
     * to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DownloadFileResult> containing the information
     * and content returned when downloading from a file.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::DownloadFileResult> Download(
        const DownloadFileOptions& options = DownloadFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Creates a new file, or updates the content of an existing file. Updating
     * an existing file overwrites any existing metadata on the file.
     * @param buffer A memory buffer containing the content to upload.
     * @param bufferSize Size of the memory buffer.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<UploadFileFromResult> containing the information
     * returned when uploading a file from a buffer.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::UploadFileFromResult> UploadFrom(
        const uint8_t* buffer,
        size_t bufferSize,
        const UploadFileFromOptions& options = UploadFileFromOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Creates a new file, or updates the content of an existing file. Updating
     * an existing file overwrites any existing metadata on the file.
     * @param fileName A file containing the content to upload.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::UploadFileFromResult> containing the
     * information returned when uploading a file from a local file.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::UploadFileFromResult> UploadFrom(
        const std::string& fileName,
        const UploadFileFromOptions& options = UploadFileFromOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Downloads a file or a file range from the service to a memory buffer using parallel
     * requests.
     * @param buffer A memory buffer to write the file content to.
     * @param bufferSize Size of the memory buffer. Size must be larger or equal to size of the file
     * or file range.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DownloadFileToResult> containing the
     * information returned when downloading a file to a local buffer.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::DownloadFileToResult> DownloadTo(
        uint8_t* buffer,
        size_t bufferSize,
        const DownloadFileToOptions& options = DownloadFileToOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Downloads a file or a file range from the service to a file using parallel
     * requests.
     * @param fileName A file path to write the downloaded content to.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DownloadFileToResult> containing the
     * information returned when downloading a file to a local file.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::DownloadFileToResult> DownloadTo(
        const std::string& fileName,
        const DownloadFileToOptions& options = DownloadFileToOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Schedules the file for deletion.
     * @param expiryOrigin Specify the origin of expiry.
     * @param options Optional parameters to schedule the file for deletion.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::ScheduleFileDeletionResult> containing the
     * information and content returned when schedule the file for deletion.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::ScheduleFileDeletionResult> ScheduleDeletion(
        ScheduleFileExpiryOriginType expiryOrigin,
        const ScheduleFileDeletionOptions& options = ScheduleFileDeletionOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Returns the result of a query against the file.
     *
     * @param querySqlExpression The query expression in SQL.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A QueryFileResult describing the query result.
     */
    Azure::Response<Models::QueryFileResult> Query(
        const std::string& querySqlExpression,
        const QueryFileOptions& options = QueryFileOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

  private:
    explicit DataLakeFileClient(
        Azure::Core::Url fileUrl,
        Blobs::BlobClient blobClient,
        std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> pipeline,
        _detail::DatalakeClientConfiguration clientConfiguration)
        : DataLakePathClient(
            std::move(fileUrl),
            std::move(blobClient),
            pipeline,
            std::move(clientConfiguration))
    {
    }

    friend class DataLakeFileSystemClient;
    friend class DataLakeDirectoryClient;
  };
}}}} // namespace Azure::Storage::Files::DataLake
