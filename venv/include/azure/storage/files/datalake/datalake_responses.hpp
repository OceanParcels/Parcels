// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_options.hpp"

#include <azure/storage/blobs/blob_responses.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  class DataLakeServiceClient;
  class DataLakeFileSystemClient;
  class DataLakePathClient;
  class DataLakeDirectoryClient;

  namespace Models {

    using LeaseDurationType = Blobs::Models::LeaseDurationType;
    using LeaseDuration [[deprecated]] = LeaseDurationType;
    using LeaseState = Blobs::Models::LeaseState;
    using LeaseStatus = Blobs::Models::LeaseStatus;

    // ServiceClient models:

    using UserDelegationKey = Blobs::Models::UserDelegationKey;
    using RetentionPolicy = Blobs::Models::RetentionPolicy;
    using AnalyticsLogging = Blobs::Models::AnalyticsLogging;
    using Metrics = Blobs::Models::Metrics;
    using CorsRule = Blobs::Models::CorsRule;
    using StaticWebsite = Blobs::Models::StaticWebsite;
    using DataLakeServiceProperties = Blobs::Models::BlobServiceProperties;
    using SetServicePropertiesResult = Blobs::Models::SetServicePropertiesResult;

    /**
     * @brief The detailed information of a file system.
     */
    struct FileSystemItemDetails final
    {
      /**
       * An HTTP entity tag associated with the file system.
       */
      Azure::ETag ETag;

      /**
       * The data and time the file system was last modified.
       */
      Azure::DateTime LastModified;

      /**
       * The Metadata of the file system.
       */
      Storage::Metadata Metadata;

      /**
       * The public access type of the file system.
       */
      PublicAccessType AccessType = PublicAccessType::None;

      /**
       * A boolean that indicates if the file system has immutability policy.
       */
      bool HasImmutabilityPolicy = false;

      /**
       * A boolean that indicates if the file system has legal hold.
       */
      bool HasLegalHold = false;

      /**
       * The duration of the lease on the file system if it has one.
       */
      Azure::Nullable<Models::LeaseDurationType> LeaseDuration;

      /**
       * The lease state of the file system.
       */
      Models::LeaseState LeaseState = Models::LeaseState::Available;

      /**
       * The lease status of the file system.
       */
      Models::LeaseStatus LeaseStatus = Models::LeaseStatus::Unlocked;

      /**
       * The default encryption scope for the file system.
       */
      std::string DefaultEncryptionScope = "$account-encryption-key";

      /**
       * Indicates whether the filesystem's default encryption scope can be overriden.
       */
      bool PreventEncryptionScopeOverride = false;
    }; // struct FileSystemItemDetails

    /**
     * @brief The file system item returned when listing the file systems.
     */
    struct FileSystemItem final
    {
      /**
       * The name of the file system.
       */
      std::string Name;

      /**
       * The detailed information of the file system.
       */
      FileSystemItemDetails Details;
    }; // struct BlobContainerItem

    // FileSystemClient models:

    /**
     * @brief The access policy of a file system.
     */
    struct FileSystemAccessPolicy final
    {
      /**
       * The public access type of the file system.
       */
      PublicAccessType AccessType = PublicAccessType::None;

      /**
       * The signed identifiers of the file system.
       */
      std::vector<SignedIdentifier> SignedIdentifiers;
    };

    using SetFileSystemAccessPolicyResult = Blobs::Models::SetBlobContainerAccessPolicyResult;

    /**
     * @brief The properties of a file system.
     */
    struct FileSystemProperties final
    {
      /**
       * An HTTP entity tag associated with the file system.
       */
      Azure::ETag ETag;

      /**
       * The data and time the file system was last modified.
       */
      DateTime LastModified;

      /**
       * The Metadata of the file system.
       */
      Storage::Metadata Metadata;

      /**
       * The default encryption scope for the file system.
       */
      std::string DefaultEncryptionScope = "$account-encryption-key";

      /**
       * Indicates whether the filesystem's default encryption scope can be overriden.
       */
      bool PreventEncryptionScopeOverride = false;
    };

    /**
     * @brief The information returned when creating the file system.
     */
    struct CreateFileSystemResult final
    {
      /**
       * If the object is created.
       */
      bool Created = true;

      /**
       * An HTTP entity tag associated with the file system.
       */
      Azure::ETag ETag;

      /**
       * The data and time the file system was last modified.
       */
      DateTime LastModified;
    };

    /**
     * @brief The information returned when deleting the file system.
     */
    struct DeleteFileSystemResult final
    {
      /**
       * If the object is deleted.
       */
      bool Deleted = true;
    };

    /**
     * @brief The information returned when setting the filesystem's metadata
     */
    struct SetFileSystemMetadataResult final
    {
      /**
       * An HTTP entity tag associated with the file system.
       */
      Azure::ETag ETag;

      /**
       * The data and time the file system was last modified.
       */
      DateTime LastModified;
    };

    // PathClient models:

    using AcquireLeaseResult = Blobs::Models::AcquireLeaseResult;
    using RenewLeaseResult = Blobs::Models::RenewLeaseResult;
    using ReleaseLeaseResult = Blobs::Models::ReleaseLeaseResult;
    using ChangeLeaseResult = Blobs::Models::ChangeLeaseResult;
    using BreakLeaseResult = Blobs::Models::BreakLeaseResult;
    using RehydratePriority = Blobs::Models::RehydratePriority;
    using ArchiveStatus = Blobs::Models::ArchiveStatus;

    /**
     * @brief The path item returned when listing the paths.
     */
    struct PathItem final
    {
      /**
       * The name of the path.
       */
      std::string Name;

      /**
       * Indicates whether this path is a directory.
       */
      bool IsDirectory = false;

      /**
       * The data and time the path was last modified.
       */
      DateTime LastModified;

      /**
       * The size of the file.
       */
      int64_t FileSize = int64_t();

      /**
       * The owner of the path.
       */
      std::string Owner;

      /**
       * The group of the path.
       */
      std::string Group;

      /**
       * The permission of the path.
       */
      std::string Permissions;

      /**
       * The name of the encryption scope under which the blob is encrypted.
       */
      Nullable<std::string> EncryptionScope;

      /**
       * Encryption context of the file. Encryption context is metadata that is not encrypted when
       * stored on the file. The primary application of this field is to store non-encrypted data
       * that can be used to derive the customer-provided key for a file.
       * Not applicable for directories.
       */
      Nullable<std::string> EncryptionContext;

      /**
       * The creation time of the path.
       */
      Nullable<DateTime> CreatedOn;

      /**
       * The expiry time of the path.
       */
      Nullable<DateTime> ExpiresOn;

      /**
       * An HTTP entity tag associated with the path.
       */
      std::string ETag;
    };

    /**
     * @brief The properties of the path.
     */
    struct PathProperties final
    {
      /**
       * An HTTP entity tag associated with the path.
       */
      Azure::ETag ETag;

      /**
       * The data and time the path was last modified.
       */
      DateTime LastModified;

      /**
       * The date and time at which the path was created.
       */
      DateTime CreatedOn;

      /**
       * The size of the file.
       */
      int64_t FileSize = 0;

      /**
       * The metadata of the path.
       */
      Storage::Metadata Metadata;

      /**
       * The duration of the lease on the path.
       */
      Azure::Nullable<Models::LeaseDurationType> LeaseDuration;

      /**
       * The state of the lease on the path.
       */
      Azure::Nullable<Models::LeaseState> LeaseState;

      /**
       * The status of the lease on the path.
       */
      Azure::Nullable<Models::LeaseStatus> LeaseStatus;

      /**
       * The common HTTP headers of the path.
       */
      PathHttpHeaders HttpHeaders;

      /**
       * A boolean indicates if the server is encrypted.
       */
      Azure::Nullable<bool> IsServerEncrypted;

      /**
       * The encryption key's SHA256.
       */
      Azure::Nullable<std::vector<uint8_t>> EncryptionKeySha256;

      /**
       * Returns the name of the encryption scope used to encrypt the path contents and application
       * metadata.  Note that the absence of this header implies use of the default account
       * encryption scope.
       */
      Nullable<std::string> EncryptionScope;

      /**
       * Encryption context of the file. Encryption context is metadata that is not encrypted when
       * stored on the file. The primary application of this field is to store non-encrypted data
       * that can be used to derive the customer-provided key for a file.
       * Not applicable for directories.
       */
      Nullable<std::string> EncryptionContext;

      /**
       * The copy ID of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyId;

      /**
       * The copy source of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<std::string> CopySource;

      /**
       * The copy status of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<Blobs::Models::CopyStatus> CopyStatus;

      /**
       * The copy progress of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyProgress;

      /**
       * The copy completion time of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<DateTime> CopyCompletedOn;

      /**
       * The expiry time of the path.
       */
      Azure::Nullable<DateTime> ExpiresOn;

      /**
       * The time this path is last accessed on.
       */
      Azure::Nullable<DateTime> LastAccessedOn;

      /**
       * A boolean indicates if the path is a directory.
       */
      bool IsDirectory = false;

      /**
       * The archive status of the path.
       */
      Azure::Nullable<Models::ArchiveStatus> ArchiveStatus;

      /**
       * The rehydrate priority of the path.
       */
      Azure::Nullable<Models::RehydratePriority> RehydratePriority;

      /**
       * The copy status's description of the path, if the path is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyStatusDescription;

      /**
       * A boolean indicates if the path has been incremental copied.
       */
      Azure::Nullable<bool> IsIncrementalCopy;

      /**
       * The incremental copy destination snapshot of the path.
       */
      Azure::Nullable<std::string> IncrementalCopyDestinationSnapshot;

      /**
       * The version ID of the path.
       */
      Azure::Nullable<std::string> VersionId;

      /**
       * A boolean indicates if the path is in its current version.
       */
      Azure::Nullable<bool> IsCurrentVersion;

      /**
       * The acls of the path.
       */
      Azure::Nullable<std::vector<Acl>> Acls;

      /**
       * The owner of the path.
       */
      Azure::Nullable<std::string> Owner;

      /**
       * The owning group of the path.
       */
      Azure::Nullable<std::string> Group;

      /**
       * The permissions of the path.
       */
      Azure::Nullable<std::string> Permissions;
    };

    /**
     * @brief The access control list of a path.
     */
    struct PathAccessControlList final
    {
      /**
       * The owner of the path.
       */
      std::string Owner;

      /**
       * The group of the path.
       */
      std::string Group;

      /**
       * The permission of the path.
       */
      std::string Permissions;

      /**
       * The acls of the path.
       */
      std::vector<Acl> Acls;
    };

    /**
     * @brief The information returned when setting the path's HTTP headers.
     */
    struct SetPathHttpHeadersResult final
    {
      /**
       * An HTTP entity tag associated with the path.
       */
      Azure::ETag ETag;

      /**
       * The data and time the path was last modified.
       */
      DateTime LastModified;
    };

    /**
     * @brief The information returned when setting the path's metadata.
     */
    struct SetPathMetadataResult final
    {
      /**
       * An HTTP entity tag associated with the path.
       */
      Azure::ETag ETag;

      /**
       * The data and time the path was last modified.
       */
      DateTime LastModified;
    };

    using SetPathPermissionsResult = SetPathAccessControlListResult;

    /**
     * @brief A path that has been soft deleted.
     */
    struct PathDeletedItem final
    {
      /**
       * The name of the path.
       */
      std::string Name;

      /**
       * The deletion ID associated with the deleted path.
       */
      std::string DeletionId;

      /**
       * When the path was deleted.
       */
      DateTime DeletedOn;

      /**
       * The number of days left before the soft deleted path will be permanently deleted.
       */
      int64_t RemainingRetentionDays = int64_t();
    };

    // FileClient models:

    using UploadFileFromResult = Blobs::Models::UploadBlockBlobResult;
    using ScheduleFileDeletionResult = Blobs::Models::SetBlobExpiryResult;
    using CopyStatus = Blobs::Models::CopyStatus;

    /**
     * @brief Response type for #Azure::Storage::Files::DataLake::DataLakeFileClient::Query.
     */
    struct QueryFileResult final
    {
      /** The response body stream.
       */
      std::unique_ptr<Core::IO::BodyStream> BodyStream;
      /**
       * Returns the date and time the container was last modified. Any operation that modifies the
       * file, including an update of the file's metadata or properties, changes the last-modified
       * time of the file.
       */
      DateTime LastModified;
      /**
       * The ETag contains a value that you can use to perform operations conditionally. If the
       * request version is 2011-08-18 or newer, the ETag value will be in quotes.
       */
      Azure::ETag ETag;
      /**
       * When a file is leased, specifies whether the lease is of infinite or fixed duration.
       */
      Nullable<LeaseDurationType> LeaseDuration;
      /**
       * Lease state of the file.
       */
      Models::LeaseState LeaseState;
      /**
       * The current lease status of the file.
       */
      Models::LeaseStatus LeaseStatus;
      /**
       * The value of this header is set to true if the file data and application metadata are
       * completely encrypted using the specified algorithm. Otherwise, the value is set to false
       * (when the file is unencrypted, or if only parts of the file/application metadata are
       * encrypted).
       */
      bool IsServerEncrypted = bool();
    };

    /**
     * @brief The detailed information returned when downloading a file.
     */
    struct DownloadFileDetails final
    {
      /**
       * An HTTP entity tag associated with the file.
       */
      Azure::ETag ETag;

      /**
       * The data and time the file was last modified.
       */
      DateTime LastModified;

      /**
       * The lease duration of the file.
       */
      Azure::Nullable<Models::LeaseDurationType> LeaseDuration;

      /**
       * The lease state of the file.
       */
      Models::LeaseState LeaseState;

      /**
       * The lease status of the file.
       */
      Models::LeaseStatus LeaseStatus;

      /**
       * The common HTTP headers of the file.
       */
      PathHttpHeaders HttpHeaders;

      /**
       * The metadata of the file.
       */
      Storage::Metadata Metadata;

      /**
       * The time this file is created on.
       */
      DateTime CreatedOn;

      /**
       * The time this file expires on.
       */
      Azure::Nullable<DateTime> ExpiresOn;

      /**
       * The time this file is last accessed on.
       */
      Azure::Nullable<DateTime> LastAccessedOn;

      /**
       * The copy ID of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyId;

      /**
       * The copy source of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<std::string> CopySource;

      /**
       * The copy status of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<Models::CopyStatus> CopyStatus;

      /**
       * The copy status's description of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyStatusDescription;

      /**
       * The copy progress of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<std::string> CopyProgress;

      /**
       * The copy completed time of the file, if the file is created from a copy operation.
       */
      Azure::Nullable<Azure::DateTime> CopyCompletedOn;

      /**
       * The version ID of the file.
       */
      Azure::Nullable<std::string> VersionId;

      /**
       * If the file is in its current version.
       */
      Azure::Nullable<bool> IsCurrentVersion;

      /**
       * A boolean indicates if the service is encrypted.
       */
      bool IsServerEncrypted = false;

      /**
       * The encryption key's SHA256.
       */
      Azure::Nullable<std::vector<uint8_t>> EncryptionKeySha256;

      /**
       * The encryption scope.
       */
      Azure::Nullable<std::string> EncryptionScope;

      /**
       * Encryption context of the file. Encryption context is metadata that is not encrypted when
       * stored on the file. The primary application of this field is to store non-encrypted data
       * that can be used to derive the customer-provided key for a file.
       * Not applicable for directories.
       */
      Nullable<std::string> EncryptionContext;

      /**
       * The acls of the file.
       */
      Azure::Nullable<std::vector<Acl>> Acls;

      /**
       * The owner of the file.
       */
      Azure::Nullable<std::string> Owner;

      /**
       * The owning group of the file.
       */
      Azure::Nullable<std::string> Group;

      /**
       * The permissions of the file.
       */
      Azure::Nullable<std::string> Permissions;
    };

    /**
     * @brief The content and information returned when downloading a file.
     */
    struct DownloadFileResult final
    {
      /**
       * The body of the downloaded result.
       */
      std::unique_ptr<Azure::Core::IO::BodyStream> Body;

      /**
       * The size of the file.
       */
      int64_t FileSize = int64_t();

      /**
       * The range of the downloaded content.
       */
      Azure::Core::Http::HttpRange ContentRange;

      /**
       * The transactional hash of the downloaded content.
       */
      Azure::Nullable<Storage::ContentHash> TransactionalContentHash;

      /**
       * The detailed information of the downloaded file.
       */
      DownloadFileDetails Details;
    };

    /**
     * @brief The information returned when downloading a file to a specific destination.
     */
    struct DownloadFileToResult final
    {
      /**
       * The size of the file.
       */
      int64_t FileSize = int64_t();

      /**
       * The range of the downloaded content.
       */
      Azure::Core::Http::HttpRange ContentRange;

      /**
       * The detailed information of the downloaded file.
       */
      DownloadFileDetails Details;
    };

    using CreateFileResult = CreatePathResult;
    using DeleteFileResult = DeletePathResult;

    // DirectoryClient models:

    using CreateDirectoryResult = CreatePathResult;
    using DeleteDirectoryResult = DeletePathResult;

  } // namespace Models

  /**
   * @brief Response type for
   * #Azure::Storage::Files::DataLake::DataLakeServiceClient::ListFileSystems.
   */
  class ListFileSystemsPagedResponse final
      : public Azure::Core::PagedResponse<ListFileSystemsPagedResponse> {
  public:
    /**
     * Service endpoint.
     */
    std::string ServiceEndpoint;
    /**
     * File system name prefix that's used to filter the result.
     */
    std::string Prefix;
    /**
     * File system items.
     */
    std::vector<Models::FileSystemItem> FileSystems;

  private:
    void OnNextPage(const Azure::Core::Context& context);

    std::shared_ptr<DataLakeServiceClient> m_dataLakeServiceClient;
    ListFileSystemsOptions m_operationOptions;

    friend class DataLakeServiceClient;
    friend class Azure::Core::PagedResponse<ListFileSystemsPagedResponse>;
  };

  /**
   * @brief Response type for #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::ListPaths
   * and #Azure::Storage::Files::DataLake::DataLakeDirectoryClient::ListPaths.
   */
  class ListPathsPagedResponse final : public Azure::Core::PagedResponse<ListPathsPagedResponse> {
  public:
    /**
     * Path items.
     */
    std::vector<Models::PathItem> Paths;

  private:
    void OnNextPage(const Azure::Core::Context& context);

    std::shared_ptr<DataLakeFileSystemClient> m_fileSystemClient;
    std::shared_ptr<DataLakeDirectoryClient> m_directoryClient;
    bool m_recursive = false;
    ListPathsOptions m_operationOptions;

    friend class DataLakeFileSystemClient;
    friend class DataLakeDirectoryClient;
    friend class Azure::Core::PagedResponse<ListPathsPagedResponse>;
  };

  /**
   * @brief Response type for
   * #Azure::Storage::Files::DataLake::DataLakeFileSystemClient::ListDeletedPaths.
   */
  class ListDeletedPathsPagedResponse final
      : public Azure::Core::PagedResponse<ListDeletedPathsPagedResponse> {
  public:
    /**
     * Deleted path items.
     */
    std::vector<Models::PathDeletedItem> DeletedPaths;

  private:
    void OnNextPage(const Azure::Core::Context& context);

    std::shared_ptr<DataLakeFileSystemClient> m_fileSystemClient;
    ListDeletedPathsOptions m_operationOptions;

    friend class DataLakeFileSystemClient;
    friend class Azure::Core::PagedResponse<ListDeletedPathsPagedResponse>;
  };

  /**
   * @brief Response type for
   * #Azure::Storage::Files::DataLake::DataLakePathClient::SetAccessControlListRecursive.
   */
  class SetPathAccessControlListRecursivePagedResponse final
      : public Azure::Core::PagedResponse<SetPathAccessControlListRecursivePagedResponse> {
  public:
    /**
     * Number of directories where Access Control List has been updated successfully.
     */
    int32_t NumberOfSuccessfulDirectories = 0;
    /**
     * Number of files where Access Control List has been updated successfully.
     */
    int32_t NumberOfSuccessfulFiles = 0;
    /**
     * Number of paths where Access Control List update has failed.
     */
    int32_t NumberOfFailures = 0;
    /**
     * A collection of path entries that failed to update ACL.
     */
    std::vector<Models::AclFailedEntry> FailedEntries;

  private:
    void OnNextPage(const Azure::Core::Context& context);

    std::shared_ptr<DataLakePathClient> m_dataLakePathClient;
    SetPathAccessControlListRecursiveOptions m_operationOptions;
    std::vector<Models::Acl> m_acls;
    Models::_detail::PathSetAccessControlListRecursiveMode m_mode;

    friend class DataLakePathClient;
    friend class Azure::Core::PagedResponse<SetPathAccessControlListRecursivePagedResponse>;
  };

  using UpdatePathAccessControlListRecursivePagedResponse
      = SetPathAccessControlListRecursivePagedResponse;
  using RemovePathAccessControlListRecursivePagedResponse
      = SetPathAccessControlListRecursivePagedResponse;

}}}} // namespace Azure::Storage::Files::DataLake
