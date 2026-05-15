import argparse
import asyncio
import time

import aiohttp

from atomization.stages.extract_blog import generate_blog_post
from atomization.stages.extract_ideas import extract_ideas_from_blog
from atomization.stages.rate_ideas import rate_and_refine_ideas
from atomization.utils.core import (
    format_time,
    get_paper_completion_status,
    load_existing_output,
    load_papers_from_file,
)
from atomization.utils.cost import (
    ensure_cost_tracking,
    format_cost,
    get_total_cost,
    reset_cost_tracker,
)
from atomization.utils.pdf import compute_truncation_stats
from atomization.storage.backend import StorageBackend

MAX_CONCURRENT_DOWNLOADS = 10


async def process_single_paper(
    paper_id: str,
    pdf_url: str,
    session: aiohttp.ClientSession,
    blog_semaphore: asyncio.Semaphore,
    idea_semaphore: asyncio.Semaphore,
    refinement_semaphore: asyncio.Semaphore,
    http_semaphore: asyncio.Semaphore,
    storage: StorageBackend,
    stats: dict,
    max_pages: int | None,
) -> None:
    """Process a single paper with resume capability: generate blog, extract ideas, refine."""
    status = get_paper_completion_status(paper_id, storage.base_dir)

    if all(status.values()):
        print(f"[{paper_id}] Fully processed, skipping.")
        stats["skipped"] += 1
        return

    blog_content = None

    if status["blog"]:
        print(f"[{paper_id}] Blog exists, loading from disk...")
        blog_content = load_existing_output(paper_id, "blog", storage.base_dir)
        stats["successful_blogs"] += 1
    else:
        try:
            print(f"[{paper_id}] Starting blog generation...")
            async with blog_semaphore:
                blog_result = await generate_blog_post(
                    pdf_url,
                    paper_id,
                    session,
                    storage,
                    max_pages,
                    http_semaphore,
                )

            if not blog_result.get("success"):
                print(
                    f"[{paper_id}] Blog failed: "
                    f"{blog_result.get('error', 'Unknown error')}"
                )
                return

            blog_content = blog_result.get("content")
            storage_loc = blog_result["storage"].get("local")
            print(f"[{paper_id}] Blog created: {storage_loc}")
            stats["successful_blogs"] += 1

            if blog_result.get("truncation"):
                trunc = blog_result["truncation"]
                stats["truncation_data"].append(trunc)

        except Exception as e:
            print(f"[{paper_id}] Blog exception: {e}")
            return

    ideas_json = None

    if status["ideas"]:
        print(f"[{paper_id}] Ideas exist, loading from disk...")
        ideas_json = load_existing_output(paper_id, "ideas", storage.base_dir)
        stats["successful_ideas"] += 1
    else:
        try:
            print(f"[{paper_id}] Starting idea extraction...")
            async with idea_semaphore:
                idea_result = await extract_ideas_from_blog(paper_id, blog_content, storage)

            if not idea_result.get("success"):
                print(
                    f"[{paper_id}] Ideas failed: "
                    f"{idea_result.get('error', 'Unknown error')}"
                )
                return

            ideas_json = idea_result.get("ideas_json")
            storage_loc = idea_result["storage"].get("local")
            print(
                f"[{paper_id}] Ideas extracted: {storage_loc} "
                f"({idea_result['count']} ideas)"
            )
            stats["successful_ideas"] += 1

        except Exception as e:
            print(f"[{paper_id}] Ideas exception: {e}")
            return

    if status["refined_ideas"]:
        print(f"[{paper_id}] Refined ideas exist, skipping refinement.")
        stats["successful_refinements"] += 1
    else:
        try:
            print(f"[{paper_id}] Starting idea refinement...")
            async with refinement_semaphore:
                refinement_result = await rate_and_refine_ideas(paper_id, ideas_json, storage)

            if not refinement_result.get("success"):
                print(
                    f"[{paper_id}] Refinement failed: "
                    f"{refinement_result.get('error', 'Unknown error')}"
                )
                return

            storage_loc = refinement_result["storage"].get("local")
            print(
                f"[{paper_id}] Ideas refined: {storage_loc} "
                f"({refinement_result['count']} ideas rated)"
            )
            stats["successful_refinements"] += 1

        except Exception as e:
            print(f"[{paper_id}] Refinement exception: {e}")


async def process_papers(
    papers: list[tuple[str, str]],
    storage: StorageBackend,
    max_concurrent_blogs: int = 50,
    max_concurrent_ideas: int = 50,
    max_concurrent_refinements: int = 50,
    max_pages: int | None = 30,
) -> None:
    """Process all papers with controlled concurrency."""
    pipeline_start = time.time()

    ensure_cost_tracking()
    reset_cost_tracker()

    print(f"Starting pipeline with {len(papers)} papers...")
    print(f"  Output directory: {storage.base_dir}")
    print(f"  Max PDF pages: {max_pages if max_pages else 'unlimited'}")
    print(f"  Max concurrent blogs: {max_concurrent_blogs}")
    print(f"  Max concurrent ideas: {max_concurrent_ideas}")
    print(f"  Max concurrent refinements: {max_concurrent_refinements}")
    print(f"  Max concurrent PDF downloads: {MAX_CONCURRENT_DOWNLOADS}")
    print()

    stats = {
        "successful_blogs": 0,
        "successful_ideas": 0,
        "successful_refinements": 0,
        "skipped": 0,
        "truncation_data": [],
    }
    blog_semaphore = asyncio.Semaphore(max_concurrent_blogs)
    idea_semaphore = asyncio.Semaphore(max_concurrent_ideas)
    refinement_semaphore = asyncio.Semaphore(max_concurrent_refinements)
    http_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    print("=" * 60)
    print("PROCESSING")
    print("=" * 60)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [
            asyncio.create_task(
                process_single_paper(
                    paper_id,
                    pdf_url,
                    session,
                    blog_semaphore,
                    idea_semaphore,
                    refinement_semaphore,
                    http_semaphore,
                    storage,
                    stats,
                    max_pages,
                )
            )
            for paper_id, pdf_url in papers
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    pipeline_elapsed = time.time() - pipeline_start
    total_cost = get_total_cost()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total papers in input: {len(papers)}")
    print(f"Papers skipped (fully processed): {stats['skipped']}")
    print(f"Blog posts created: {stats['successful_blogs']}")
    print(f"Ideas extracted: {stats['successful_ideas']}")
    print(f"Ideas refined: {stats['successful_refinements']}")

    if stats["truncation_data"]:
        truncation_stats = compute_truncation_stats(stats["truncation_data"])
        print("\nPDF Truncation Summary:")
        print(f"  Total pages before: {truncation_stats['total_original_pages']}")
        print(f"  Total pages after: {truncation_stats['total_kept_pages']}")
        print(
            f"  Pages removed: {truncation_stats['total_removed_pages']} "
            f"({truncation_stats['removal_percentage']:.1f}%)"
        )
        print(
            f"  Avg pages per paper: {truncation_stats['avg_original_pages']:.1f} "
            f"-> {truncation_stats['avg_kept_pages']:.1f}"
        )
        print("  Truncation reasons:")
        for reason, count in truncation_stats["reasons"].items():
            print(f"    {reason}: {count}")

    print(f"\nTotal time: {format_time(pipeline_elapsed)}")
    print(f"Total cost: {format_cost(total_cost)}")


def run_pipeline() -> None:
    parser = argparse.ArgumentParser(
        description="Process research papers through blog generation and atom extraction."
    )
    parser.add_argument("papers_file", help="Path to TSV file: paper_id<TAB>pdf_url")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="papers",
        help="Output directory for generated files (default: papers)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process from the file (default: all)",
    )
    parser.add_argument(
        "--max-concurrent-blogs",
        type=int,
        default=50,
        help="Maximum concurrent blog generation tasks (default: 50)",
    )
    parser.add_argument(
        "--max-concurrent-ideas",
        type=int,
        default=50,
        help="Maximum concurrent idea extraction tasks (default: 50)",
    )
    parser.add_argument(
        "--max-concurrent-refinements",
        type=int,
        default=50,
        help="Maximum concurrent idea refinement tasks (default: 50)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum PDF pages to process (default: 30, use 0 for unlimited)",
    )

    args = parser.parse_args()

    storage = StorageBackend(base_dir=args.output_dir)

    papers = load_papers_from_file(args.papers_file)

    if not papers:
        print("Error: No valid papers found in the file.")
        parser.exit(1)

    if args.max_papers is not None and args.max_papers > 0:
        papers = papers[:args.max_papers]
        print(f"Processing first {len(papers)} papers from {args.papers_file}")

    max_pages = args.max_pages if args.max_pages > 0 else None

    asyncio.run(
        process_papers(
            papers,
            storage,
            max_concurrent_blogs=args.max_concurrent_blogs,
            max_concurrent_ideas=args.max_concurrent_ideas,
            max_concurrent_refinements=args.max_concurrent_refinements,
            max_pages=max_pages,
        )
    )
