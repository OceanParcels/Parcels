// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/blobs/rest_client.hpp"
#include "azure/storage/files/datalake/rest_client.hpp"

#include <azure/core/nullable.hpp>
#include <azure/storage/blobs/blob_options.hpp>
#include <azure/storage/common/access_conditions.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  namespace Models {
    using PathHttpHeaders = Blobs::Models::BlobHttpHeaders;
    using ListFileSystemsIncludeFlags = Blobs::Models::ListBlobContainersIncludeFlags;
    using SignedIdentifier = Blobs::Models::SignedIdentifier;
    using FileQueryArrowField = Blobs::Models::BlobQueryArrowField;
    using FileQueryArrowFieldType = Blobs::Models::BlobQueryArrowFieldType;
    using EncryptionAlgorithmType = Blobs::Models::EncryptionAlgorithmType;

    /**
     * @brief An access control object.
     */
    struct Acl final
    {
      /**
       * The scope of the ACL.
       */
      std::string Scope;

      /**
       * The type of the ACL.
       */
      std::string Type;

      /**
       * The ID of the ACL.
       */
      std::string Id;

      /**
       * The permissions of the ACL.
       */
      std::string Permissions;

      /**
       * @brief Creates an Acl based on acl input string.
       * @param aclString the string to be parsed to Acl.
       * @return Acl
       */
      static Acl FromString(const std::string& aclString);

      /**
       * @brief Creates a string from an Acl.
       * @param acl the acl object to be serialized to a string.
       * @return std::string
       */
      static std::string ToString(const Acl& acl);

      /**
       * @brief Creates a vector of Acl from a string that indicates multiple acls.
       * @param aclsString the string that contains multiple acls.
       * @return std::vector<Acl>
       */
      static std::vector<Acl> DeserializeAcls(const std::string& aclsString);

      /**
       * @brief Creates a string that contains several Acls.
       * @param aclsArray the acls to be serialized into a string.
       * @return std::string
       */
      static std::string SerializeAcls(const std::vector<Acl>& aclsArray);
    };
  } // namespace Models

  using DownloadFileToOptions = Blobs::DownloadBlobToOptions;
  using GetUserDelegationKeyOptions = Blobs::GetUserDelegationKeyOptions;
  using GetServicePropertiesOptions = Blobs::GetServicePropertiesOptions;
  using SetServicePropertiesOptions = Blobs::SetServicePropertiesOptions;
  using EncryptionKey = Blobs::EncryptionKey;

  namespace _detail {
    struct DatalakeClientConfiguration
    {

      /**
       * API version used by this client.
       */
      std::string ApiVersion;

      /**
       * @brief The token credential used to initialize the client.
       */
      std::shared_ptr<Core::Credentials::TokenCredential> TokenCredential;

      /**
       * @brief Holds the customer provided key used when making requests.
       */
      Azure::Nullable<EncryptionKey> CustomerProvidedKey;

      /**
       * The filesystem url. This is only non-null for directory clients that are created from a
       * filesystem client, so that this directory client knows where to send ListPaths requests.
       */
      Azure::Nullable<Azure::Core::Url> FileSystemUrl;
    };
  } // namespace _detail

  /**
   * @brief Audiences available for data lake service
   *
   */
  class DataLakeAudience final
      : public Azure::Core::_internal::ExtendableEnumeration<DataLakeAudience> {
  public:
    /**
     * @brief Construct a new DataLakeAudience object
     *
     * @param dataLakeAudience The Azure Active Directory audience to use when forming
     * authorization scopes. For the Language service, this value corresponds to a URL that
     * identifies the Azure cloud where the resource is located. For more information: See
     * https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-access-azure-active-directory
     */
    explicit DataLakeAudience(std::string dataLakeAudience)
        : ExtendableEnumeration(std::move(dataLakeAudience))
    {
    }

    /**
     * @brief The service endpoint for a given storage account. Use this method to acquire a token
     * for authorizing requests to that specific Azure Storage account and service only.
     *
     * @param storageAccountName he storage account name used to populate the service endpoint.
     * @return The service endpoint for a given storage account.
     */
    static DataLakeAudience CreateDataLakeServiceAccountAudience(
        const std::string& storageAccountName)
    {
      return DataLakeAudience("https://" + storageAccountName + ".blob.core.windows.net/");
    }

    /**
     * @brief Default Audience. Use to acquire a token for authorizing requests to any Azure
     * Storage account.
     */
    AZ_STORAGE_FILES_DATALAKE_DLLEXPORT const static DataLakeAudience DefaultAudience;
  };

  /**
   * @brief Client options used to initialize all DataLake clients.
   */
  struct DataLakeClientOptions final : Azure::Core::_internal::ClientOptions
  {
    /**
     * SecondaryHostForRetryReads specifies whether the retry policy should retry a read
     * operation against another host. If SecondaryHostForRetryReads is "" (the default) then
     * operations are not retried against another host. NOTE: Before setting this field, make sure
     * you understand the issues around reading stale & potentially-inconsistent data at this
     * webpage: https://docs.microsoft.com/azure/storage/common/geo-redundant-design.
     */
    std::string SecondaryHostForRetryReads;

    /**
     * API version used by this client.
     */
    std::string ApiVersion;

    /**
     * @brief Holds the customer provided key used when making requests.
     */
    Azure::Nullable<EncryptionKey> CustomerProvidedKey;

    /**
     * Enables tenant discovery through the authorization challenge when the client is configured to
     * use a TokenCredential. When enabled, the client will attempt an initial un-authorized request
     * to prompt a challenge in order to discover the correct tenant for the resource.
     */
    bool EnableTenantDiscovery = false;

    /**
     * The Audience to use for authentication with Azure Active Directory (AAD).
     * #Azure::Storage::Files::DataLake::DataLakeAudience::DefaultAudience will be assumed
     * if Audience is not set.
     */
    Azure::Nullable<DataLakeAudience> Audience;
  };

  /**
   * @brief Specifies access conditions for a file system.
   */
  struct FileSystemAccessConditions final : public Azure::ModifiedConditions,
                                            public LeaseAccessConditions
  {
  };

  /**
   * @brief Specifies access conditions for a path.
   */
  struct PathAccessConditions final : public Azure::ModifiedConditions,
                                      public Azure::MatchConditions,
                                      public LeaseAccessConditions
  {
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeServiceClient::ListFileSystems.
   */
  struct ListFileSystemsOptions final
  {
    /**
     * Filters results to filesystems within the specified prefix.
     */
    Azure::Nullable<std::string> Prefix;

    /**
     * The number of filesystems returned with each invocation is limited. If the number of
     * filesystems to be returned exceeds this limit, a continuation token is returned in the
     * response header x-ms-continuation. When a continuation token is returned in the response, it
     * must be specified in a subsequent invocation of the list operation to continue listing the
     * filesystems.
     */
    Azure::Nullable<std::string> ContinuationToken;

    /**
     * An optional value that specifies the maximum number of items to return. If omitted or greater
     * than 5,000, the response will include up to 5,000 items.
     */
    Azure::Nullable<int32_t> PageSizeHint;

    /**
     * Specifies that the filesystem's metadata be returned.
     */
    Models::ListFileSystemsIncludeFlags Include = Models::ListFileSystemsIncludeFlags::None;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::Create.
   */
  struct CreateFileSystemOptions final
  {
    /**
     * User-defined metadata to be stored with the filesystem. Note that the string may only contain
     * ASCII characters in the ISO-8859-1 character set.
     */
    Storage::Metadata Metadata;

    /**
     * The public access type of the file system.
     */
    Models::PublicAccessType AccessType = Models::PublicAccessType::None;

    /**
     * @brief The encryption scope to use as the default on the filesystem.
     */
    Azure::Nullable<std::string> DefaultEncryptionScope;

    /**
     * @brief If true, prevents any file upload from specifying a different encryption
     * scope.
     */
    Azure::Nullable<bool> PreventEncryptionScopeOverride;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::Delete.
   */
  struct DeleteFileSystemOptions final
  {
    /**
     * Specify the access condition for the file system.
     */
    FileSystemAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::GetProperties.
   */
  struct GetFileSystemPropertiesOptions final
  {
    /**
     * Specify the lease access conditions.
     */
    LeaseAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::SetMetadata.
   */
  struct SetFileSystemMetadataOptions final
  {
    struct : public LeaseAccessConditions
    {
      /**
       * @brief Specify this header to perform the operation only if the resource has been
       * modified since the specified time. This timestamp will be truncated to second.
       */
      Azure::Nullable<Azure::DateTime> IfModifiedSince;
    } /**
       * Specify the access condition for the file system.
       */
    AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::ListPaths.
   */
  struct ListPathsOptions final
  {
    /**
     * Valid only when Hierarchical Namespace is enabled for the account. If "true", the user
     * identity values returned in the owner and group fields of each list entry will be transformed
     * from Azure Active Directory Object IDs to User Principal Names. If "false" or not provided,
     * the values will be returned as Azure Active Directory Object IDs. Note that group and
     * application Object IDs are not translated because they do not have unique friendly names.
     * More Details about UserPrincipalName, See
     * https://learn.microsoft.com/entra/identity/hybrid/connect/plan-connect-userprincipalname#what-is-userprincipalname
     */
    Azure::Nullable<bool> UserPrincipalName;

    /**
     * The number of paths returned with each invocation is limited. If the number of paths to be
     * returned exceeds this limit, a continuation token is returned in the response header
     * x-ms-continuation. When a continuation token is returned in the response, it must be
     * specified in a subsequent invocation of the list operation to continue listing the paths.
     */
    Azure::Nullable<std::string> ContinuationToken;

    /**
     * An optional value that specifies the maximum number of items to return. If omitted or greater
     * than 5,000, the response will include up to 5,000 items.
     */
    Azure::Nullable<int32_t> PageSizeHint;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::ListPaths.
   */
  struct ListDeletedPathsOptions final
  {
    /**
     * Gets the paths that have recently been soft deleted in this file system.
     */
    Azure::Nullable<std::string> Prefix;

    /**
     * The number of paths returned with each invocation is limited. If the number of paths to be
     * returned exceeds this limit, a continuation token is returned in the response header
     * x-ms-continuation. When a continuation token is returned in the response, it must be
     * specified in a subsequent invocation of the list operation to continue listing the paths.
     */
    Azure::Nullable<std::string> ContinuationToken;

    /**
     * An optional value that specifies the maximum number of items to return. If omitted or greater
     * than 5,000, the response will include up to 5,000 items.
     */
    Azure::Nullable<int32_t> PageSizeHint;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::UndeletePath.
   */
  struct UndeletePathOptions final
  {
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::GetAccessPolicy.
   */
  struct GetFileSystemAccessPolicyOptions final
  {
    /**
     * Optional conditions that must be met to perform this operation.
     */
    LeaseAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::SetAccessPolicy.
   */
  struct SetFileSystemAccessPolicyOptions final
  {
    /**
     * Specifies whether data in the file system may be accessed publicly and the level of access.
     */
    Models::PublicAccessType AccessType = Models::PublicAccessType::None;

    /**
     * Stored access policies that you can use to provide fine grained control over file system
     * permissions.
     */
    std::vector<Models::SignedIdentifier> SignedIdentifiers;

    /**
     * Optional conditions that must be met to perform this operation.
     */
    FileSystemAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::RenameDirectory.
   * @remark Some optional parameter is mandatory in certain combination.
   *         More details:
   * https://docs.microsoft.com/rest/api/storageservices/datalakestoragegen2/path/create
   */
  struct RenameDirectoryOptions final
  {
    /**
     * If not specified, the source's file system is used. Otherwise, rename to destination file
     * system.
     */
    Azure::Nullable<std::string> DestinationFileSystem;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * The access condition for source path.
     */
    PathAccessConditions SourceAccessConditions;
  };

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakeFileClient::Append.
   */
  struct AppendFileOptions final
  {
    /**
     * Specify the transactional hash for the body, to be validated by the service.
     */
    Azure::Nullable<Storage::ContentHash> TransactionalContentHash;

    /**
     * Specify the lease access conditions.
     */
    LeaseAccessConditions AccessConditions;

    /**
     * If true, the file will be flushed after the append.
     */
    Azure::Nullable<bool> Flush;

    /**
     * If "acquire" it will acquire the lease.
     * If "auto-renew" it will renew the lease.
     * If "release" it will release the lease only on flush. Only applicable if Flush is set to
     * true.
     * If "acquire-release" it will acquire & complete the operation & release the lease once
     * operation is done. Only applicable if Flush is set to true.
     */
    Azure::Nullable<Models::LeaseAction> LeaseAction;

    /**
     * Proposed LeaseId.
     */
    Azure::Nullable<std::string> LeaseId;

    /**
     * Specifies the duration of the lease, in seconds, or InfiniteLeaseDuration for a lease that
     * never expires. A non-infinite lease can be between 15 and 60 seconds. A lease duration cannot
     * be changed using renew or change.
     */
    Azure::Nullable<std::chrono::seconds> LeaseDuration;
  };

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakeFileClient::Flush.
   */
  struct FlushFileOptions final
  {
    /**
     * If "true", uncommitted data is retained after the flush operation completes; otherwise, the
     * uncommitted data is deleted after the flush operation.  The default is false.  Data at
     * offsets less than the specified position are written to the file when flush succeeds, but
     * this optional parameter allows data after the flush position to be retained for a future
     * flush operation.
     */
    Azure::Nullable<bool> RetainUncommittedData;

    /**
     * Azure Storage Events allow applications to receive notifications when files change. When
     * Azure Storage Events are enabled, a file changed event is raised. This event has a property
     * indicating whether this is the final change to distinguish the difference between an
     * intermediate flush to a file stream and the final close of a file stream. The close query
     * parameter is valid only when the action is "flush" and change notifications are enabled. If
     * the value of close is "true" and the flush operation completes successfully, the service
     * raises a file change notification with a property indicating that this is the final update
     * (the file stream has been closed). If "false" a change notification is raised indicating the
     * file has changed. The default is false. This query parameter is set to true by the Hadoop
     * ABFS driver to indicate that the file stream has been closed."
     */
    Azure::Nullable<bool> Close;

    /**
     * The service stores this value and is returned for "Read & Get Properties" operations. If this
     * property is not specified on the request, then the property will be cleared for the file.
     * Subsequent calls to "Read & Get Properties" will not return this property unless it is
     * explicitly set on that file again.
     */
    Azure::Nullable<Storage::ContentHash> ContentHash;

    /**
     * Specify the HTTP headers for this path.
     */
    Models::PathHttpHeaders HttpHeaders;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * If "acquire" it will acquire the lease.
     * If "auto-renew" it will renew the lease.
     * If "release" it will release the lease only on flush. Only applicable if Flush is set to
     * true.
     * If "acquire-release" it will acquire & complete the operation & release the lease once
     * operation is done. Only applicable if Flush is set to true.
     */
    Azure::Nullable<Models::LeaseAction> LeaseAction;

    /**
     * Proposed LeaseId.
     */
    Azure::Nullable<std::string> LeaseId;

    /**
     * Specifies the duration of the lease, in seconds, or InfiniteLeaseDuration for a lease that
     * never expires. A non-infinite lease can be between 15 and 60 seconds. A lease duration cannot
     * be changed using renew or change.
     */
    Azure::Nullable<std::chrono::seconds> LeaseDuration;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::SetAccessControlList.
   */
  struct SetPathAccessControlListOptions final
  {
    /**
     * The owner of the path or directory.
     */
    Azure::Nullable<std::string> Owner;

    /**
     * The owning group of the path or directory.
     */
    Azure::Nullable<std::string> Group;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::SetPermissions.
   */
  struct SetPathPermissionsOptions final
  {
    /**
     * The owner of the path or directory.
     */
    Azure::Nullable<std::string> Owner;

    /**
     * The owning group of the path or directory.
     */
    Azure::Nullable<std::string> Group;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileClient::SetHttpHeaders.
   */
  struct SetPathHttpHeadersOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::SetMetadata.
   */
  struct SetPathMetadataOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  using ScheduleFileExpiryOriginType = Blobs::Models::ScheduleBlobExpiryOriginType;

  /**
   * @brief Options for scheduling the deletion of a path.
   */
  struct ScheduleFileDeletionOptions final
  {
    /**
     * The expiry time from the specified origin. Only work if ExpiryOrigin is
     * ScheduleFileExpiryOriginType::RelativeToCreation or
     * ScheduleFileExpiryOriginType::RelativeToNow.
     * Does not apply to directories.
     * TimeToExpire and ExpiresOn cannot both be set.
     */
    Azure::Nullable<std::chrono::milliseconds> TimeToExpire;

    /**
     * The expiry time in RFC1123 format. Only works if ExpiryOrigin is
     * ScheduleFileExpiryOriginType::Absolute.
     * Does not apply to directories.
     * ExpiresOn and TimeToExpire cannot both be set.
     */
    Azure::Nullable<DateTime> ExpiresOn;
  };

  using SchedulePathDeletionOptions = ScheduleFileDeletionOptions;

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakePathClient::Create.
   * @remark Some optional parameter is mandatory in certain combination.
   *         More details:
   * https://docs.microsoft.com/rest/api/storageservices/datalakestoragegen2/path/create
   */
  struct CreatePathOptions final
  {
    /**
     * Specify the HTTP headers for this path.
     */
    Models::PathHttpHeaders HttpHeaders;

    /**
     * User-defined metadata to be stored with the path. Note that the string may only contain ASCII
     * characters in the ISO-8859-1 character set.  If the filesystem exists, any metadata not
     * included in the list will be removed.  All metadata are removed if the header is omitted.  To
     * merge new and existing metadata, first get all existing metadata and the current E-Tag, then
     * make a conditional request with the E-Tag and include values for all metadata.
     */
    Storage::Metadata Metadata;

    /**
     * Only valid if Hierarchical Namespace is enabled for the account. When creating a file or
     * directory and the parent folder does not have a default ACL, the umask restricts the
     * permissions of the file or directory to be created.  The resulting permission is given by p
     * bitwise and not u, where p is the permission and u is the umask.  For example, if p is 0777
     * and u is 0057, then the resulting permission is 0720.  The default permission is 0777 for a
     * directory and 0666 for a file. The default umask is 0027.  The umask must be specified in
     * 4-digit octal notation (e.g. 0766).
     */
    Azure::Nullable<std::string> Umask;

    /**
     * Only valid if Hierarchical Namespace is enabled for the account. Sets POSIX access
     * permissions for the file owner, the file owning group, and others. Each class may be granted
     * read, write, or execute permission. The sticky bit is also supported.  Both symbolic
     * (rwxrw-rw-) and 4-digit octal notation (e.g. 0766) are supported.
     */
    Azure::Nullable<std::string> Permissions;

    /**
     * The owner of the file or directory.
     */
    Azure::Nullable<std::string> Owner;

    /**
     * The owning group of the file or directory.
     */
    Azure::Nullable<std::string> Group;

    /**
     * Sets POSIX access control rights on files and directories. Each access control entry (ACE)
     * consists of a scope, a type, a user or group identifier, and permissions.
     */
    Azure::Nullable<std::vector<Models::Acl>> Acls;

    /**
     * Proposed LeaseId.
     */
    Azure::Nullable<std::string> LeaseId;

    /**
     * Specifies the duration of the lease, in seconds, or InfiniteLeaseDuration for a lease that
     * never expires. A non-infinite lease can be between 15 and 60 seconds. A lease duration cannot
     * be changed using renew or change. Does not apply to directories.
     */
    Azure::Nullable<std::chrono::seconds> LeaseDuration;

    /**
     * Optional parameters to schedule the file for deletion.
     */
    SchedulePathDeletionOptions ScheduleDeletionOptions;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * Encryption context of the file. Encryption context is metadata that is not encrypted when
     * stored on the file. The primary application of this field is to store non-encrypted data that
     * can be used to derive the customer-provided key for a file.
     * Not applicable for directories.
     */
    Azure::Nullable<std::string> EncryptionContext;
  };

  /**
   * @brief Optional parameters for Azure::Storage::Files::DataLake::DirectoryClient::Delete.
   * @remark Some optional parameter is mandatory in certain combination.
   *         More details:
   * https://docs.microsoft.com/rest/api/storageservices/datalakestoragegen2/path/delete
   */
  struct DeletePathOptions final
  {
    /**
     * Required and valid only when the resource is a directory. If "true", all paths beneath the
     * directory will be deleted. If "false" and the directory is non-empty, an error occurs.
     */
    Azure::Nullable<bool> Recursive;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::GetProperties.
   * @remark Some optional parameter is mandatory in certain combination.
   *         More details:
   * https://docs.microsoft.com/rest/api/storageservices/datalakestoragegen2/path/getproperties
   */
  struct GetPathPropertiesOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * Valid only when Hierarchical Namespace is enabled for the account. If "true", the user
     * identity values returned in the owner and group fields of each list entry will be transformed
     * from Azure Active Directory Object IDs to User Principal Names. If "false" or not provided,
     * the values will be returned as Azure Active Directory Object IDs. Note that group and
     * application Object IDs are not translated because they do not have unique friendly names.
     * More Details about UserPrincipalName, See
     * https://learn.microsoft.com/entra/identity/hybrid/connect/plan-connect-userprincipalname#what-is-userprincipalname
     */
    Nullable<bool> IncludeUserPrincipalName;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::GetAccessControlList.
   */
  struct GetPathAccessControlListOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * Valid only when Hierarchical Namespace is enabled for the account. If "true", the user
     * identity values returned in the owner and group fields of each list entry will be transformed
     * from Azure Active Directory Object IDs to User Principal Names. If "false" or not provided,
     * the values will be returned as Azure Active Directory Object IDs. Note that group and
     * application Object IDs are not translated because they do not have unique friendly names.
     * More Details about UserPrincipalName, See
     * https://learn.microsoft.com/entra/identity/hybrid/connect/plan-connect-userprincipalname#what-is-userprincipalname
     */
    Nullable<bool> IncludeUserPrincipalName;
  };

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakeFileClient::Download.
   * @remark Some optional parameter is mandatory in certain combination.
   *         More details:
   * https://docs.microsoft.com/rest/api/storageservices/datalakestoragegen2/path/read
   */
  struct DownloadFileOptions final
  {
    /**
     * Specify the range of the resource to be retrieved.
     */
    Azure::Nullable<Core::Http::HttpRange> Range;

    /**
     * The hash algorithm used to calculate the hash for the returned content.
     */
    Azure::Nullable<HashAlgorithm> RangeHashAlgorithm;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * Valid only when Hierarchical Namespace is enabled for the account. If "true", the user
     * identity values returned in the owner and group fields of each list entry will be transformed
     * from Azure Active Directory Object IDs to User Principal Names. If "false" or not provided,
     * the values will be returned as Azure Active Directory Object IDs. Note that group and
     * application Object IDs are not translated because they do not have unique friendly names.
     * More Details about UserPrincipalName, See
     * https://learn.microsoft.com/entra/identity/hybrid/connect/plan-connect-userprincipalname#what-is-userprincipalname
     */
    Nullable<bool> IncludeUserPrincipalName;
  };

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::RenameFile.
   */
  struct RenameFileOptions final
  {
    /**
     * If not specified, the source's file system is used. Otherwise, rename to destination file
     * system.
     */
    Azure::Nullable<std::string> DestinationFileSystem;

    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;

    /**
     * The access condition for source path.
     */
    PathAccessConditions SourceAccessConditions;
  };

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakeFileClient::Delete.
   */
  struct DeleteFileOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  using RenameSubdirectoryOptions = RenameDirectoryOptions;

  /**
   * @brief Optional parameters for
   * Azure::Storage::Files::DataLake::DataLakeDirectoryClient::Delete.
   */
  struct DeleteDirectoryOptions final
  {
    /**
     * Specify the access condition for the path.
     */
    PathAccessConditions AccessConditions;
  };

  /**
   * @brief Optional parameters for DirectoryClient::SetAccessControlListRecursive.
   */
  struct SetPathAccessControlListRecursiveOptions final
  {
    /**
     * When performing setAccessControlRecursive on a directory, the number of paths that are
     * processed with each invocation is limited.  If the number of paths to be processed exceeds
     * this limit, a continuation token is returned in this response header.  When a continuation
     * token is returned in the response, it must be specified in a subsequent invocation of the
     * setAccessControlRecursive operation to continue the setAccessControlRecursive operation on
     * the directory.
     */
    Azure::Nullable<std::string> ContinuationToken;

    /**
     * It specifies the maximum number of files or directories on which the acl change will be
     * applied. If omitted or greater than 2,000, the request will process up to 2,000 items.
     */
    Azure::Nullable<int32_t> PageSizeHint;

    /**
     * Optional. If set to false, the operation will terminate quickly on encountering user
     * errors (4XX). If true, the operation will ignore user errors and proceed with the operation
     * on other sub-entities of the directory. Continuation token will only be returned when
     * ContinueOnFailure is true in case of user errors. If not set the default value is false for
     * this.
     */
    Azure::Nullable<bool> ContinueOnFailure;
  };

  using UpdatePathAccessControlListRecursiveOptions = SetPathAccessControlListRecursiveOptions;

  using RemovePathAccessControlListRecursiveOptions = SetPathAccessControlListRecursiveOptions;

  using CreateFileOptions = CreatePathOptions;
  using CreateDirectoryOptions = CreatePathOptions;

  /**
   * @brief Optional parameters for
   * #Azure::Storage::Files::DataLake::DataLakeFileClient::UploadFrom.
   */
  struct UploadFileFromOptions final
  {
    /**
     * The standard HTTP header system properties to set.
     */
    Models::PathHttpHeaders HttpHeaders;

    /**
     * Name-value pairs associated with the blob as metadata.
     */
    Storage::Metadata Metadata;

    /**
     * Options for parallel transfer.
     */
    struct
    {
      /**
       * File smaller than this will be uploaded with a single upload operation. This value
       * cannot be larger than 5000 MiB.
       */
      int64_t SingleUploadThreshold = 256 * 1024 * 1024;

      /**
       * The maximum number of bytes in a single request. This value cannot be larger than
       * 4000 MiB.
       */
      Azure::Nullable<int64_t> ChunkSize;

      /**
       * The maximum number of threads that may be used in a parallel transfer.
       */
      int32_t Concurrency = 5;
    } TransferOptions;
  };

  using AcquireLeaseOptions = Blobs::AcquireLeaseOptions;
  using BreakLeaseOptions = Blobs::BreakLeaseOptions;
  using RenewLeaseOptions = Blobs::RenewLeaseOptions;
  using ReleaseLeaseOptions = Blobs::ReleaseLeaseOptions;
  using ChangeLeaseOptions = Blobs::ChangeLeaseOptions;

  using FileQueryInputTextOptions = Blobs::BlobQueryInputTextOptions;
  using FileQueryOutputTextOptions = Blobs::BlobQueryOutputTextOptions;
  using FileQueryError = Blobs::BlobQueryError;

  /**
   * @brief Optional parameters for #Azure::Storage::Files::DataLake::DataLakeFileClient::Query.
   */
  struct QueryFileOptions final
  {
    /**
     * @brief Input text configuration.
     */
    FileQueryInputTextOptions InputTextConfiguration;
    /**
     * @brief Output text configuration.
     */
    FileQueryOutputTextOptions OutputTextConfiguration;
    /**
     * @brief Optional conditions that must be met to perform this operation.
     */
    PathAccessConditions AccessConditions;
    /**
     * @brief Callback for progress handling.
     */
    std::function<void(int64_t, int64_t)> ProgressHandler;
    /**
     * @brief Callback for error handling. If you don't specify one, the default will be used, which
     * will ignore all non-fatal errors and throw for fatal errors.
     */
    std::function<void(FileQueryError)> ErrorHandler;
  };
}}}} // namespace Azure::Storage::Files::DataLake
