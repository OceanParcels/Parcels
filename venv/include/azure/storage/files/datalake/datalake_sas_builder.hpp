// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_responses.hpp"

#include <azure/core/nullable.hpp>
#include <azure/storage/common/account_sas_builder.hpp>

#include <cstdint>
#include <string>
#include <type_traits>

namespace Azure { namespace Storage { namespace Sas {

  /**
   * @brief Specifies which resources are accessible via the shared access signature.
   */
  enum class DataLakeSasResource
  {
    /**
     * @brief Grants access to the content and metadata of any files and directories in the
     * filesystem, and to the list of files and directories in the filesystem.
     */
    FileSystem,

    /**
     * @brief Grants access to the content and metadata of the file.
     */
    File,

    /**
     * @brief grants access to the files and subdirectories in the directory and to list the paths
     * in the directory.
     */
    Directory,
  };

  /**
   * @brief The list of permissions that can be set for a filesystem's access policy.
   */
  enum class DataLakeFileSystemSasPermissions
  {
    /**
     * @brief Indicates that Read is permitted.
     */
    Read = 1,

    /**
     * @brief Indicates that Write is permitted.
     */
    Write = 2,

    /**
     * @brief Indicates that Delete is permitted.
     */
    Delete = 4,

    /**
     * @brief Indicates that List is permitted.
     */
    List = 8,

    /**
     * @brief Indicates that Add is permitted.
     */
    Add = 16,

    /**
     * @brief Indicates that Create is permitted.
     */
    Create = 32,

    /**
     * @brief Indicates that all permissions are set.
     */
    All = ~0,
  };

  inline DataLakeFileSystemSasPermissions operator|(
      DataLakeFileSystemSasPermissions lhs,
      DataLakeFileSystemSasPermissions rhs)
  {
    using type = std::underlying_type_t<DataLakeFileSystemSasPermissions>;
    return static_cast<DataLakeFileSystemSasPermissions>(
        static_cast<type>(lhs) | static_cast<type>(rhs));
  }

  inline DataLakeFileSystemSasPermissions operator&(
      DataLakeFileSystemSasPermissions lhs,
      DataLakeFileSystemSasPermissions rhs)
  {
    using type = std::underlying_type_t<DataLakeFileSystemSasPermissions>;
    return static_cast<DataLakeFileSystemSasPermissions>(
        static_cast<type>(lhs) & static_cast<type>(rhs));
  }

  /**
   * @brief The list of permissions that can be set for a file or directory's access policy.
   */
  enum class DataLakeSasPermissions
  {
    /**
     * @brief Indicates that Read is permitted.
     */
    Read = 1,

    /**
     * @brief Indicates that Write is permitted.
     */
    Write = 2,

    /**
     * @brief Indicates that Delete is permitted.
     */

    Delete = 4,

    /**
     * @brief Indicates that Add is permitted.
     */
    Add = 8,

    /**
     * @brief Indicates that Create is permitted.
     */
    Create = 16,

    /**
     * @brief Indicates that List is permitted.
     */
    List = 32,

    /**
     * @brief Allows the caller to move a blob (file or directory) to a new location.
     */
    Move = 64,

    /**
     * @brief Allows the caller to get system properties and POSIX ACLs of a blob (file or
     * directory).
     */
    Execute = 128,

    /**
     * @brief Allows the caller to set owner, owning group, or act as the owner when renaming or
     * deleting a blob (file or directory) within a folder that has the sticky bit set.
     */
    ManageOwnership = 256,

    /**
     * @brief Allows the caller to set permissions and POSIX ACLs on blobs (files and directories).
     */
    ManageAccessControl = 512,

    /**
     * @brief Indicates that all permissions are set.
     */
    All = ~0,
  };

  inline DataLakeSasPermissions operator|(DataLakeSasPermissions lhs, DataLakeSasPermissions rhs)
  {
    using type = std::underlying_type_t<DataLakeSasPermissions>;
    return static_cast<DataLakeSasPermissions>(static_cast<type>(lhs) | static_cast<type>(rhs));
  }

  inline DataLakeSasPermissions operator&(DataLakeSasPermissions lhs, DataLakeSasPermissions rhs)
  {
    using type = std::underlying_type_t<DataLakeSasPermissions>;
    return static_cast<DataLakeSasPermissions>(static_cast<type>(lhs) & static_cast<type>(rhs));
  }

  /**
   * @brief DataLakeSasBuilder is used to generate a Shared Access Signature (SAS) for an Azure
   * Storage DataLake filesystem or path.
   */
  struct DataLakeSasBuilder final
  {
    /**
     * @brief The optional signed protocol field specifies the protocol permitted for a
     * request made with the SAS.
     */
    SasProtocol Protocol;

    /**
     * @brief Optionally specify the time at which the shared access signature becomes
     * valid. This timestamp will be truncated to second.
     */
    Azure::Nullable<Azure::DateTime> StartsOn;

    /**
     * @brief The time at which the shared access signature becomes invalid. This field must
     * be omitted if it has been specified in an associated stored access policy. This timestamp
     * will be truncated to second.
     */
    Azure::DateTime ExpiresOn;

    /**
     * @brief Specifies an IP address or a range of IP addresses from which to accept
     * requests. If the IP address from which the request originates does not match the IP address
     * or address range specified on the SAS token, the request is not authenticated. When
     * specifying a range of IP addresses, note that the range is inclusive.
     */
    Azure::Nullable<std::string> IPRange;

    /**
     * @brief An optional unique value up to 64 characters in length that correlates to an
     * access policy specified for the filesystem.
     */
    std::string Identifier;

    /**
     * @brief The name of the filesystem being made accessible.
     */
    std::string FileSystemName;

    /**
     * @brief The name of the path being made accessible, or empty for a filesystem SAS.
     */
    std::string Path;

    /**
     * @brief Defines whether or not the Path is a directory. If this value is set to true, the Path
     * is a directory for a directory SAS. If set to false or default, the Path is a file for a file
     * SAS.
     */
    bool IsDirectory = false;

    /**
     * @brief Required when Resource is set to Directory to indicate the depth of the directory
     * specified in the canonicalized resource field of the string-to-sign to indicate the depth of
     * the directory specified in the canonicalized resource field of the string-to-sign. This is
     * only used for user delegation SAS.
     */
    Azure::Nullable<int32_t> DirectoryDepth;

    /**
     * @brief Specifies which resources are accessible via the shared access signature.
     */
    DataLakeSasResource Resource;

    /**
     * @brief Override the value returned for Cache-Control response header.
     */
    std::string CacheControl;

    /**
     * @brief Override the value returned for Content-Disposition response header.
     */
    std::string ContentDisposition;

    /**
     * @brief Override the value returned for Content-Encoding response header.
     */
    std::string ContentEncoding;

    /**
     * @brief Override the value returned for Content-Language response header.
     */
    std::string ContentLanguage;

    /**
     * @brief Override the value returned for Content-Type response header.
     */
    std::string ContentType;

    /**
     * @brief This value will be used for the AAD Object ID of a user authorized by the owner of the
     * User Delegation Key to perform the action granted by the SAS. The Azure Storage service will
     * ensure that the owner of the user delegation key has the required permissions before granting
     * access. No additional permission check for the user specified in this value will be
     * performed. This cannot be used in conjuction with AgentObjectId. This is only used with
     * generating User Delegation SAS.
     */
    std::string PreauthorizedAgentObjectId;

    /**
     * @brief This value will be used for the AAD Object ID of a user authorized by the owner of the
     * User Delegation Key to perform the action granted by the SAS. The Azure Storage service will
     * ensure that the owner of the user delegation key has the required permissions before granting
     * access. The Azure Storage Service will perform an additional POSIX ACL check to determine if
     * the user is authorized to perform the requested operation. This cannot be used in conjuction
     * with PreauthorizedAgentObjectId. This is only used with generating User Delegation SAS.
     */
    std::string AgentObjectId;

    /**
     * @brief This value will be used for correlating the storage audit logs with the audit logs
     * used by the principal generating and distributing SAS. This is only used for User Delegation
     * SAS.
     */
    std::string CorrelationId;

    /**
     * @brief Optional encryption scope to use when sending requests authorized with this SAS url.
     */
    std::string EncryptionScope;

    /**
     * @brief Sets the permissions for the filesystem SAS.
     *
     * @param permissions The allowed permissions.
     */
    void SetPermissions(DataLakeFileSystemSasPermissions permissions);

    /**
     * @brief Sets the permissions for the file SAS or directory SAS.
     *
     * @param permissions The allowed permissions.
     */
    void SetPermissions(DataLakeSasPermissions permissions);

    /**
     * @brief Uses the StorageSharedKeyCredential to sign this shared access signature, to produce
     * the proper SAS query parameters for authentication requests.
     *
     * @param credential The storage account's shared key credential.
     * @return The SAS query parameters used for authenticating requests.
     */
    std::string GenerateSasToken(const StorageSharedKeyCredential& credential);

    /**
     * @brief Sets the permissions for the SAS using a raw permissions string.
     *
     * @param rawPermissions Raw permissions string for the SAS.
     */
    void SetPermissions(std::string rawPermissions) { Permissions = std::move(rawPermissions); }

    /**
     * @brief Uses an account's user delegation key to sign this shared access signature, to
     * produce the proper SAS query parameters for authentication requests.
     *
     * @param userDelegationKey UserDelegationKey returned from
     * BlobServiceClient.GetUserDelegationKey.
     * @param accountName The name of the storage account.
     * @return The SAS query parameters used for authenticating requests.
     */
    std::string GenerateSasToken(
        const Files::DataLake::Models::UserDelegationKey& userDelegationKey,
        const std::string& accountName);

  private:
    std::string Permissions;
  };

}}} // namespace Azure::Storage::Sas
