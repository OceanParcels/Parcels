// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_file_system_client.hpp"
#include "azure/storage/files/datalake/datalake_options.hpp"
#include "azure/storage/files/datalake/datalake_responses.hpp"

#include <azure/core/credentials/credentials.hpp>
#include <azure/core/internal/http/pipeline.hpp>
#include <azure/core/response.hpp>
#include <azure/storage/blobs/blob_client.hpp>
#include <azure/storage/common/storage_credential.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  /** @brief The DataLakePathClient allows you to manipulate Azure Storage DataLake files.
   *
   */
  class DataLakePathClient {
  public:
    /**
     * @brief Destructor.
     *
     */
    virtual ~DataLakePathClient() = default;

    /**
     * @brief Create from connection string
     * @param connectionString Azure Storage connection string.
     * @param fileSystemName The name of a file system.
     * @param path The path of a resource within the file system.
     * @param options Optional parameters used to initialize the client.
     * @return DataLakePathClient
     */
    static DataLakePathClient CreateFromConnectionString(
        const std::string& connectionString,
        const std::string& fileSystemName,
        const std::string& path,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Shared key authentication client.
     * @param pathUrl The URL of the path this client's request targets.
     * @param credential The shared key credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakePathClient(
        const std::string& pathUrl,
        std::shared_ptr<StorageSharedKeyCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Bearer token authentication client.
     * @param pathUrl The URL of the path this client's request targets.
     * @param credential The token credential used to initialize the client.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakePathClient(
        const std::string& pathUrl,
        std::shared_ptr<Core::Credentials::TokenCredential> credential,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Anonymous/SAS/customized pipeline auth.
     * @param pathUrl The URL of the path this client's request targets.
     * @param options Optional parameters used to initialize the client.
     */
    explicit DataLakePathClient(
        const std::string& pathUrl,
        const DataLakeClientOptions& options = DataLakeClientOptions());

    /**
     * @brief Gets the path's primary URL endpoint. This is the endpoint used for blob
     * storage available features in DataLake.
     *
     * @return The path's primary URL endpoint.
     */
    std::string GetUrl() const { return m_blobClient.GetUrl(); }

    /**
     * @brief Creates a file or directory. By default, the destination is overwritten and
     *        if the destination already exists and has a lease the lease is broken.
     * @param type Type of resource to create.
     * @param options Optional parameters to create the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreatePathResult> containing the information
     * returned when creating a path.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreatePathResult> Create(
        Models::PathResourceType type,
        const CreatePathOptions& options = CreatePathOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Creates a file or directory. By default, the destination is not changed if it already
     * exists.
     * @param type Type of resource to create.
     * @param options Optional parameters to create the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::CreatePathResult> containing the information
     * returned when creating a path, the information will only be valid when the create operation
     * is successful.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::CreatePathResult> CreateIfNotExists(
        Models::PathResourceType type,
        const CreatePathOptions& options = CreatePathOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the resource the path points to.
     * @param options Optional parameters to delete the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeletePathResult> which is current empty but
     * preserved for future usage.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeletePathResult> Delete(
        const DeletePathOptions& options = DeletePathOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Deletes the resource the path points to if it exists.
     * @param options Optional parameters to delete the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::DeletePathResult> which is current empty but
     * preserved for future usage. The result will only valid if the delete operation is successful.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::DeletePathResult> DeleteIfExists(
        const DeletePathOptions& options = DeletePathOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the owner, group, and access control list for a file or directory.
     *        Note that Hierarchical Namespace must be enabled for the account in order to use
     *        access control.
     * @param acls Sets POSIX access control rights on files and directories. Each access control
     *             entry (ACE) consists of a scope, a type, a user or group identifier, and
     *             permissions.
     * @param options Optional parameters to set an access control to the resource the path points
     *                to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::SetPathAccessControlListResult> containing the
     * information returned when setting path's access control.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::SetPathAccessControlListResult> SetAccessControlList(
        std::vector<Models::Acl> acls,
        const SetPathAccessControlListOptions& options = SetPathAccessControlListOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the owner, group, and permissions for a file or directory.
     *        Note that Hierarchical Namespace must be enabled for the account in order to use
     *        access control.
     * @param permissions Sets the permissions on the path
     * @param options Optional parameters to set permissions to the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::SetPathPermissionsResult> containing the
     * information returned when setting path's permissions.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::SetPathPermissionsResult> SetPermissions(
        std::string permissions,
        const SetPathPermissionsOptions& options = SetPathPermissionsOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the properties of a resource the path points to.
     * @param httpHeaders Sets the blob HTTP headers.
     * @param options Optional parameters to set the HTTP headers to the resource the path points
     * to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<SetPathHttpHeadersResult> containing the information
     * returned when setting the path's HTTP headers.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::SetPathHttpHeadersResult> SetHttpHeaders(
        Models::PathHttpHeaders httpHeaders,
        const SetPathHttpHeadersOptions& options = SetPathHttpHeadersOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Get Properties returns all system and user defined properties for a path. Get Status
     *        returns all system defined properties for a path. Get Access Control List returns the
     *        access control list for a path.
     * @param options Optional parameters to get the properties from the resource the path points
     *                to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::PathProperties> containing the
     * properties of the path.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::PathProperties> GetProperties(
        const GetPathPropertiesOptions& options = GetPathPropertiesOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Returns all access control list stored for the given path.
     * @param options Optional parameters to get the ACLs from the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::GetPathAccessControlListResult> containing the
     * access control list of the path.
     * @remark This request is sent to dfs endpoint.
     */
    Azure::Response<Models::PathAccessControlList> GetAccessControlList(
        const GetPathAccessControlListOptions& options = GetPathAccessControlListOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets the metadata of a resource the path points to.
     * @param metadata User-defined metadata to be stored with the filesystem. Note that the string
     *                 may only contain ASCII characters in the ISO-8859-1 character set.
     * @param options Optional parameters to set the metadata to the resource the path points to.
     * @param context Context for cancelling long running operations.
     * @return Azure::Response<Models::SetPathMetadataResult> containing the
     * information returned when setting the metadata.
     * @remark This request is sent to blob endpoint.
     */
    Azure::Response<Models::SetPathMetadataResult> SetMetadata(
        Storage::Metadata metadata,
        const SetPathMetadataOptions& options = SetPathMetadataOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const;

    /**
     * @brief Sets POSIX access control rights on files and directories under given directory
     * recursively.
     * @param acls Sets POSIX access control rights on files and directories. Each access control
     * entry (ACE) consists of a scope, a type, a user or group identifier, and permissions.
     * @param options Optional parameters to set an access control recursively to the resource the
     * directory points to.
     * @param context Context for cancelling long running operations.
     * @return SetPathAccessControlListRecursivePagedResponse containing summary stats of the
     * operation.
     * @remark This request is sent to dfs endpoint.
     */
    SetPathAccessControlListRecursivePagedResponse SetAccessControlListRecursive(
        const std::vector<Models::Acl>& acls,
        const SetPathAccessControlListRecursiveOptions& options
        = SetPathAccessControlListRecursiveOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return SetAccessControlListRecursiveInternal(
          Models::_detail::PathSetAccessControlListRecursiveMode::Set, acls, options, context);
    }

    /**
     * @brief Updates POSIX access control rights on files and directories under given directory
     * recursively.
     * @param acls Updates POSIX access control rights on files and directories. Each access control
     * entry (ACE) consists of a scope, a type, a user or group identifier, and permissions.
     * @param options Optional parameters to set an access control recursively to the resource the
     * directory points to.
     * @param context Context for cancelling long running operations.
     * @return UpdatePathAccessControlListRecursivePagedResponse containing summary stats of the
     * operation.
     * @remark This request is sent to dfs endpoint.
     */
    UpdatePathAccessControlListRecursivePagedResponse UpdateAccessControlListRecursive(
        const std::vector<Models::Acl>& acls,
        const UpdatePathAccessControlListRecursiveOptions& options
        = UpdatePathAccessControlListRecursiveOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return SetAccessControlListRecursiveInternal(
          Models::_detail::PathSetAccessControlListRecursiveMode::Modify, acls, options, context);
    }

    /**
     * @brief Removes POSIX access control rights on files and directories under given directory
     * recursively.
     * @param acls Removes POSIX access control rights on files and directories. Each access control
     * entry (ACE) consists of a scope, a type, a user or group identifier, and permissions.
     * @param options Optional parameters to set an access control recursively to the resource the
     * directory points to.
     * @param context Context for cancelling long running operations.
     * @return RemovePathAccessControlListRecursivePagedResponse containing summary stats of the
     * operation.
     * @remark This request is sent to dfs endpoint.
     */
    RemovePathAccessControlListRecursivePagedResponse RemoveAccessControlListRecursive(
        const std::vector<Models::Acl>& acls,
        const RemovePathAccessControlListRecursiveOptions& options
        = RemovePathAccessControlListRecursiveOptions(),
        const Azure::Core::Context& context = Azure::Core::Context()) const
    {
      return SetAccessControlListRecursiveInternal(
          Models::_detail::PathSetAccessControlListRecursiveMode::Remove, acls, options, context);
    }

  protected:
    /** @brief Url to the resource on the service */
    Azure::Core::Url m_pathUrl;

    /** @brief Blob Client for this path client. */
    Blobs::BlobClient m_blobClient;

    /** @brief Http Pipeline */
    std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> m_pipeline;

    /** @brief Client configurations*/
    _detail::DatalakeClientConfiguration m_clientConfiguration;

    /**
     * @brief Construct a new DataLakePathClient
     *
     * @param pathUrl The URL of the path represented by this client.
     * @param blobClient The BlobClient needed for blob operations performed on this path.
     * @param pipeline The HTTP pipeline for sending and receiving REST requests and responses.
     * @param clientConfiguration Client configurations
     *
     */
    explicit DataLakePathClient(
        Azure::Core::Url pathUrl,
        Blobs::BlobClient blobClient,
        std::shared_ptr<Azure::Core::Http::_internal::HttpPipeline> pipeline,
        _detail::DatalakeClientConfiguration clientConfiguration)
        : m_pathUrl(std::move(pathUrl)), m_blobClient(std::move(blobClient)),
          m_pipeline(std::move(pipeline)), m_clientConfiguration(std::move(clientConfiguration))
    {
    }

  private:
    SetPathAccessControlListRecursivePagedResponse SetAccessControlListRecursiveInternal(
        Models::_detail::PathSetAccessControlListRecursiveMode mode,
        const std::vector<Models::Acl>& acls,
        const SetPathAccessControlListRecursiveOptions& options,
        const Azure::Core::Context& context) const;

    friend class DataLakeFileSystemClient;
    friend class DataLakeLeaseClient;
  };
}}}} // namespace Azure::Storage::Files::DataLake
