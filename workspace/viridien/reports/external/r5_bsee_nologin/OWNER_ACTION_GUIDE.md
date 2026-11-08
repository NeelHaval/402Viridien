# Owner action guide: ten minutes to start

Checked: 8 November 2026. The agent did not sign in or submit a request.
The times below are planning estimates. They are not agency service times.
Account setup, profile work, and file selection can take more than ten minutes.

| Time | Owner action |
| --- | --- |
| 0–1 min | Open the [FRS start page](https://www.data.bsee.gov/Other/FileRequestSystem/Default.aspx). Have your own inbox and an MFA method ready. Open the copied pilot CSV in this folder. |
| 1–6 min | Select **Sign In**. Use your existing Login.gov account, or create one. For a new account: enter an address you control, confirm the message in your inbox, make a password of at least 12 characters, and set up MFA. Then return to FRS. Allow about 3–7 minutes for a simple new account. This estimate is not verified. |
| 6–8 min | Open [Well Data](https://www.data.bsee.gov/Other/FileRequestSystem/WellData.aspx). After sign-in, use the text-list upload. Upload [owner_pilot_filename_ids.txt](owner_pilot_filename_ids.txt). It has the 60 exact DOC_ID values from the earlier lane's pilot, one per line, with no extension. Use **Process Filename List** if the control appears. |
| 8–10 min | Check the records against the pilot CSV. Select the intended rows and use **Add to Custom Set**. Use the set controls to place the set in **Request Cart**. Check the file count and byte total. The owner can submit when the selection is correct. If setup or selection takes longer, stop at a safe point and continue later. |

The authenticated upload, profile, set, and checkout screens were not tested.
The public Well Data page states that text-list upload is enabled after sign-in.
Its page source names **Process Filename List**. This does not verify a complete
authenticated upload or a successful request.

If list upload fails, use **Filename ID** on Well Data. Enter `clog06009` as
the first pilot ID, with no `.tif`. Select **Submit Query**. Check API
`608114045801`, lease `G26313`, area `GC`, block `468`, suffix `ST00BP01`,
and the formation-evaluation title. The earlier lane checked this query.
Repeat with the other pilot IDs as needed. Do not select all files from a well.

The copied pilot has **60 unique IDs**, from **20 well groups**. Its listed
size is **1,490,473,041 bytes**, about 1.49 GB. This guide does not change
the earlier lane's ranking or its selection.

The public query fields are **Region**, **Group**, **Well File Type**,
**Area**, **Block**, **Lease**, **API**, **Filename ID**, **Type Wildcard**,
**Water Depth (feet)**, **Created Date**, **Run Date**, and **Released Date**.
Use exact Filename IDs for this pilot. Extra filters can hide a selected ID.
The CSV also keeps the well name, suffix, title, and source record number.
These fields let you check that you selected the correct file.

An old FRS/PIOS password no longer works. If you have an old profile, use
the same account address in Login.gov so that FRS can link the profile.
New users can update their FRS profile after sign-in. The exact required
profile and checkout fields are **unknown** from the public pages. No fixed
FRS account-approval delay was found. The FRS FAQ describes sign-in plus MFA;
it does not state that this service needs Login.gov identity proofing.

The verified service limits are separate from the time plan:

- FRS requests are free. The stated size limit is **25 GB per request**.
- A file-count limit, daily request limit, text-list count limit, and text-file
  size limit were not found on the checked public pages. Do not assume no limit.
- The public query says a query can take **up to one minute**.
- FRS says file preparation can take **up to 24 hours**, based on request volume.
  This is a published service statement, not a measured time or a guarantee.
- Status moves through **Requested → Processing → Published → Removed**.
- When the set is Published, use its link on
  [Request Status](https://www.data.bsee.gov/Other/FileRequestSystem/Authenticated/RequestStatus.aspx).
  The agency also sends the link to the account inbox. A SharePoint message
  may also arrive. The actual prepared-file host and archive format were not
  checked. No download URL was guessed.
- The link lasts **five days after activation**. The set is then removed.
  Do not treat five days as a period from request submission.

After download, keep the raw files and hashes. Hold them outside all data sets.
Check API, lease, well suffix, and original filenames for challenge overlap.
An exact challenge copy is source evidence only. Exclude it from conversion
and training. Keep alternate scales and versions from one well in one group.

Sources: [FRS rules](https://www.data.bsee.gov/Other/FileRequestSystem/Default.aspx),
[FRS Login.gov FAQ](https://www.data.bsee.gov/Other/FileRequestSystem/LoginGovFAQ.aspx),
[Login.gov account steps](https://login.gov/help/create-account/how-do-i-create-an-account/).
The saved responses and request times are in `sources/` and `access.jsonl`.
