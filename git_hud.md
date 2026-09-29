# GIT_HUD — Quy trình cập nhật source lên GitHub

## 1. Khi nào áp dụng tài liệu này

Khi người dùng nói một trong các câu tương đương sau:

- `thực hiện git_hud.md`
- `push theo git_hud.md`
- `cập nhật project lên GitHub theo file git_hud.md`

agent phải đọc toàn bộ file này rồi thực hiện quy trình kiểm tra, commit và push bên dưới. Yêu cầu đó được hiểu là người dùng cho phép tạo commit và push các thay đổi source hiện có của project này lên repository đã khai báo.

Không chạy pipeline, test, renderer hoặc tạo thêm kết quả quang học trong tác vụ Git, trừ khi người dùng yêu cầu riêng.

## 2. Repository cố định

- Thư mục project: `C:\test 8 maxing\hud_fan_pipeline -1`
- Remote: `https://github.com/Nguyenchan2005/HUD.git`
- Tên remote: `origin`
- Branch chính: `main`
- Git for Windows dự phòng: `C:\Program Files\Git\cmd\git.exe`
- Git author hiện tại: `Nguyenchan2005`
- Git email hiện tại: `Nguyenchan2005@users.noreply.github.com`

Source production có thẩm quyền nằm trực tiếp ở thư mục gốc project. Không lấy source từ các thư mục snapshot `CODE/`, `code1/` hoặc các thư mục kết quả chạy.

## 3. Dữ liệu tuyệt đối không được push

Các mục sau là kết quả chạy, cache, backup hoặc snapshot cục bộ và phải tiếp tục bị bỏ qua bởi `.gitignore`:

```text
multistart_runs_v5_5/
EXECUTION/
manual_flow_v5_5/
render_rollback_backups/
resuft/
resuft.zip
STEP_24/
temp_check_rt/
scratch/
performance_profiles/
CODE/
code1/
__pycache__/
.pytest_cache/
*.pyc
*.log
*.tmp
*.zip
```

Không được dùng `git add -f` để cưỡng ép thêm bất kỳ mục nào trong danh sách này.

Không được tự ý xóa các thư mục kết quả trên máy. `.gitignore` chỉ loại chúng khỏi Git, không xóa dữ liệu.

## 4. Nguyên tắc an toàn bắt buộc

1. Giữ nguyên mọi thay đổi source hiện có của người dùng.
2. Không chạy `git reset --hard`, `git clean`, `git checkout -- .` hoặc lệnh làm mất dữ liệu.
3. Không dùng `git push --force` hoặc `git push --force-with-lease`.
4. Không amend, rebase hoặc sửa lịch sử commit nếu người dùng không yêu cầu rõ ràng.
5. Không tạo empty commit khi không có thay đổi.
6. Không push token, password, API key, private key, file `.env` hoặc thông tin xác thực.
7. Không push file đơn lẻ lớn hơn `100 MB`.
8. Phải xem danh sách staged trước khi commit.
9. Nếu remote sai, có merge conflict, non-fast-forward, mất xác thực hoặc phát hiện dữ liệu nhạy cảm thì dừng và báo người dùng; không tự cưỡng ép giải quyết.
10. Repository hiện có thể là public. Chỉ push source và dữ liệu đầu vào mà người dùng đã đặt trong phạm vi project; không đưa kết quả chạy vào.

## 5. Xác định lệnh Git

Trong PowerShell:

```powershell
Set-Location -LiteralPath "C:\test 8 maxing\hud_fan_pipeline -1"

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $gitCommand) {
    $git = $gitCommand.Source
} else {
    $git = "C:\Program Files\Git\cmd\git.exe"
}

if (-not (Test-Path -LiteralPath $git)) {
    throw "GIT_EXECUTABLE_NOT_FOUND"
}

& $git --version
```

## 6. Kiểm tra repository trước khi stage

Chạy:

```powershell
& $git rev-parse --is-inside-work-tree
& $git branch --show-current
& $git remote -v
& $git status --short
& $git diff --check
& $git diff
```

Kết quả bắt buộc:

- Đang ở Git worktree.
- Branch hiện tại là `main`.
- `origin` trỏ tới `https://github.com/Nguyenchan2005/HUD.git`.
- `git diff --check` không báo lỗi whitespace nghiêm trọng.

Nếu chưa có `origin`, được phép thêm đúng remote:

```powershell
& $git remote add origin "https://github.com/Nguyenchan2005/HUD.git"
```

Nếu `origin` tồn tại nhưng trỏ sang URL khác, phải dừng và hỏi người dùng; không tự thay đổi remote.

## 7. Xác nhận kết quả chạy vẫn bị ignore

Chạy:

```powershell
& $git check-ignore -v `
    multistart_runs_v5_5 `
    EXECUTION `
    manual_flow_v5_5 `
    render_rollback_backups `
    resuft.zip
```

Các mục tồn tại trong project phải được `git check-ignore` xác nhận là ignored. Nếu một mục kết quả không còn bị ignore, phải sửa `.gitignore` trước khi stage.

## 8. Đồng bộ thông tin remote an toàn

Chạy fetch, không tự merge hoặc rebase:

```powershell
& $git fetch origin
```

Kiểm tra quan hệ giữa local và remote:

```powershell
& $git status -sb
```

Nếu local báo `behind`, `diverged` hoặc push sau đó bị `non-fast-forward`, dừng và báo người dùng. Không force-push.

## 9. Stage thay đổi

Nếu người dùng chỉ định rõ file cần push, stage đúng các file đó. Nếu người dùng yêu cầu cập nhật toàn bộ project, chạy:

```powershell
& $git add --all
```

Sau đó bắt buộc kiểm tra:

```powershell
& $git diff --cached --name-status
& $git diff --cached --stat
& $git diff --cached --check
```

Danh sách staged không được chứa đường dẫn thuộc mục 3. Nếu chứa, bỏ stage bằng:

```powershell
& $git restore --staged -- "DUONG_DAN_KHONG_DUOC_PUSH"
```

Không xóa file local khi bỏ stage.

Trước khi commit, kiểm tra các file mới để chắc chắn không chứa token, password, API key hoặc private key. Nếu phát hiện chuỗi đáng ngờ, dừng và báo người dùng.

## 10. Tạo commit

Nếu không có staged changes:

```powershell
& $git diff --cached --quiet
```

thì không tạo commit; báo rằng repository không có thay đổi cần push.

Nếu có thay đổi, dùng commit message mô tả đúng nội dung. Ví dụ:

```powershell
& $git commit -m "Fix STEP11 M2 topology pipeline"
```

Không dùng message mơ hồ như `update` nếu có thể mô tả thay đổi chính xác hơn.

Nếu Git chưa có author trong repository này, cấu hình cục bộ:

```powershell
& $git config user.name "Nguyenchan2005"
& $git config user.email "Nguyenchan2005@users.noreply.github.com"
```

## 11. Push

Push branch `main` theo kiểu fast-forward bình thường:

```powershell
& $git push origin main
```

Nếu GitHub yêu cầu xác thực, cho phép người dùng hoàn tất đăng nhập bằng trình duyệt hoặc Git Credential Manager. Không yêu cầu người dùng gửi token vào hội thoại.

## 12. Xác minh sau push

Chạy:

```powershell
& $git fetch origin
& $git status --short
& $git log -1 --oneline --decorate
& $git rev-parse HEAD
& $git rev-parse origin/main
```

Hoàn thành chỉ khi:

- `git status --short` không còn thay đổi chưa commit, trừ các thay đổi mới phát sinh ngoài phạm vi trong lúc agent làm việc.
- `HEAD` bằng `origin/main`.
- Push không báo lỗi.

## 13. Báo cáo cuối cùng cho người dùng

Báo ngắn gọn đủ các nội dung:

- Push thành công hay chưa.
- Commit hash và commit message.
- Số file đã thay đổi.
- Branch và remote.
- Xác nhận các thư mục kết quả vẫn bị bỏ qua.
- Nếu không push được, ghi nguyên nhân chính xác và không tuyên bố hoàn thành.

## 14. Quy trình rút gọn chuẩn

```text
Đọc git_hud.md
    ↓
Kiểm tra repo/branch/remote
    ↓
Kiểm tra diff và dữ liệu nhạy cảm
    ↓
Xác nhận kết quả chạy đang ignored
    ↓
git fetch origin
    ↓
git add --all
    ↓
Kiểm tra staged files
    ↓
git commit -m "Mô tả chính xác"
    ↓
git push origin main
    ↓
Xác nhận HEAD == origin/main
```
